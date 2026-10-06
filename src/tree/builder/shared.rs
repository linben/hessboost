//! State and arithmetic every builder shares: the configuration each builder
//! derives from its parameters, XGBoost's leaf-weight and node gain formulas,
//! interaction-constraint state, root sums, and the final leaf-value pass.

use std::collections::BTreeSet;

use crate::config::TrainingParams;
use crate::objective::GradPair;
use crate::tree::RegTree;
use crate::tree::constraints::{Bounds, MonotoneConstraints, calc_weight_bounded};
use crate::tree::gain::{GradStats, RegParams, threshold_l1};
use crate::tree::hist::{SumOrder, sum_order};
use rayon::prelude::*;

/// Whether the rayon pool has more than one thread, so parallelism can pay off.
pub(super) fn rayon_available() -> bool {
    rayon::current_num_threads() > 1
}

/// Training rows that reached a leaf during tree construction.
pub(crate) struct LeafRows {
    pub node: usize,
    pub rows: Vec<u32>,
}

/// What every builder derives from its training configuration.
pub(super) struct BuilderConfig<'a> {
    pub(super) params: &'a TrainingParams,
    pub(super) reg: RegParams,
    pub(super) cons: MonotoneConstraints,
    /// The interaction constraint groups, each sorted and deduplicated.
    /// `None` means interaction constraints are inactive (no filtering). An
    /// unlisted feature may only interact with itself.
    interaction_sets: Option<Vec<Vec<u32>>>,
}

impl<'a> BuilderConfig<'a> {
    pub(super) fn new(params: &'a TrainingParams) -> Self {
        let groups = &params.interaction_constraints;
        let interaction_sets = (!groups.is_empty()).then(|| {
            groups
                .iter()
                .map(|group| {
                    let mut group = group.clone();
                    group.sort_unstable();
                    group.dedup();
                    group
                })
                .collect()
        });
        BuilderConfig {
            params,
            reg: RegParams::from_params(params),
            cons: MonotoneConstraints::from_params(&params.monotone_constraints),
            interaction_sets,
        }
    }

    /// The interaction state both children of a split on `feature` share.
    /// XGBoost permits every feature already used on the path plus every
    /// member of a constraint group containing the *entire* updated path.
    pub(super) fn next_allowed(
        &self,
        parent: Option<&InteractionState>,
        feature: u32,
    ) -> Option<InteractionState> {
        let groups = self.interaction_sets.as_deref()?;
        let mut path = parent.map_or_else(Vec::new, |state| state.path.clone());
        if let Err(pos) = path.binary_search(&feature) {
            path.insert(pos, feature);
        }
        let mut allowed: BTreeSet<u32> = path.iter().copied().collect();
        for group in groups {
            if path
                .iter()
                .all(|feature| group.binary_search(feature).is_ok())
            {
                allowed.extend(group.iter().copied());
            }
        }
        Some(InteractionState {
            path,
            allowed: allowed.into_iter().collect(),
        })
    }
}

/// XGBoost's `CalcWeight` in `f64`: `−Tα(G)/(H+λ)`, `0` without positive
/// Hessian, clamped to `max_delta_step` when set.
#[inline]
pub(crate) fn xgb_calc_weight(stats: GradStats, reg: &RegParams) -> f64 {
    if stats.hess <= 0.0 {
        return 0.0;
    }
    let mut w = -threshold_l1(stats.grad, reg.alpha) / (stats.hess + reg.lambda);
    if reg.max_delta_step != 0.0 && w.abs() > reg.max_delta_step {
        w = reg.max_delta_step.copysign(w);
    }
    w
}

/// XGBoost's `ApplyBounds`: `w` clamped to `[lower, upper]` (a NaN passes
/// through).
#[inline(always)]
pub(super) fn apply_bounds(w: f32, lower: f32, upper: f32) -> f32 {
    if w < lower {
        lower
    } else if w > upper {
        upper
    } else {
        w
    }
}

/// XGBoost's `SplitEvaluator::CalcWeight`: [`xgb_calc_weight`] rounded to
/// `f32`, then clamped to the node's monotone bounds. The `f32` rounding
/// happens before bounding, exactly as upstream.
#[inline]
pub(super) fn xgb_weight(stats: GradStats, reg: &RegParams, bounds: Bounds) -> f32 {
    let w = xgb_calc_weight(stats, reg) as f32;
    apply_bounds(w, bounds.lower as f32, bounds.upper as f32)
}

/// XGBoost's `CalcGainGivenWeight` with an `f32` weight: `−(2Gw + (H+λ)w² +
/// 2α|w|)` where `w²` is formed in `f32` (upstream `Sqr(float)`) and every
/// other operation runs in `f64`.
#[inline]
pub(super) fn xgb_gain_given_weight(stats: GradStats, reg: &RegParams, w: f32) -> f64 {
    -(2.0 * stats.grad * f64::from(w)
        + (stats.hess + reg.lambda) * f64::from(w * w)
        + 2.0 * reg.alpha * f64::from(w.abs()))
}

/// XGBoost's scalar `TreeEvaluator::CalcGain` for a node: the given-weight
/// gain at the `f32` (bounded) weight, rounded to `f32` as upstream stores
/// `root_gain`. The histogram and exact updaters both use this form, so the
/// parent baseline carries the same `f32` weight rounding as every candidate.
pub(super) fn xgb_node_gain(stats: GradStats, reg: &RegParams, bounds: Bounds) -> f32 {
    if stats.hess <= 0.0 {
        return 0.0;
    }
    xgb_gain_given_weight(stats, reg, xgb_weight(stats, reg, bounds)) as f32
}

/// XGBoost interaction-constraint state for one node: every split feature on
/// the root-to-node path and the features still permitted there.
#[derive(Clone)]
pub(super) struct InteractionState {
    path: Vec<u32>,
    allowed: Vec<u32>,
}

/// Whether `feature` is permitted at a node. `None` means constraints inactive
/// or the root (where every feature is allowed).
pub(super) fn permits(state: Option<&InteractionState>, feature: u32) -> bool {
    state.is_none_or(|state| state.allowed.binary_search(&feature).is_ok())
}

/// Sum the gradient pairs of `rows` in the histograms' order
/// ([`sum_order`]): a node below 8,192 rows in row order, a larger one in
/// blocks, each a chain from zero, the block totals added in block order to
/// the first. The blocks depend on the row count alone, so the parallel and
/// serial sums agree (and a GPU reproduces them). Shared by the builders'
/// root-statistics accumulation.
pub(crate) fn sum_rows(gpair: &[GradPair], rows: &[u32]) -> GradStats {
    let SumOrder::Blocked { grain } = sum_order(rows.len()) else {
        return chain_rows(gpair, rows);
    };
    let partials: Vec<GradStats> = if rayon_available() {
        rows.par_chunks(grain)
            .map(|block| chain_rows(gpair, block))
            .collect()
    } else {
        rows.chunks(grain)
            .map(|block| chain_rows(gpair, block))
            .collect()
    };
    let mut partials = partials.into_iter();
    let mut total = partials.next().unwrap_or_default();
    for partial in partials {
        total.add(partial);
    }
    total
}

/// The row-order chain of `rows`' gradient pairs, from zero.
///
/// Kept out of line: inlined into `HistTreeBuilder::build_inner`, LLVM kept
/// the running sum in the caller's stack slot and paid a store-to-load round
/// trip per row.
#[inline(never)]
fn chain_rows(gpair: &[GradPair], rows: &[u32]) -> GradStats {
    let mut total = GradStats::default();
    for &r in rows {
        total.add(GradStats::from_pair(gpair[r as usize]));
    }
    total
}

/// Set every leaf's weight from its stored statistics, respecting each leaf's
/// monotone bounds. Shared by the builders' final pass.
pub(super) fn finalize_leaf_values(
    tree: &mut RegTree,
    stats: &[GradStats],
    bounds: &[Bounds],
    reg: &RegParams,
) {
    let n_nodes = tree.num_nodes();
    for (id, (&stats, &bounds)) in stats[..n_nodes].iter().zip(&bounds[..n_nodes]).enumerate() {
        if tree.node(id).is_leaf() {
            let w = calc_weight_bounded(stats, reg, bounds);
            tree.set_leaf_value(id, w as f32);
        }
    }
}
