//! Boosting rounds for `multi_strategy = multi_output_tree`: one vector-leaf
//! tree per round fits every output at once (XGBoost's `IsVectorLeaf` path),
//! for `gbtree` and DART, optionally growing its structure from reduced split
//! gradients supplied by the objective ([`Loss::split_gradient`]).

use super::dart::{dart_new_tree_weight, finish_dart, round_gradients};
use super::margins::{MarginCaches, TreeOutput};
use super::prepare::TrainContext;
use super::round::tree_eta;
use super::row_sampling::{gradient_sampling, iteration_row_subsets, make_column_sampler};
use crate::config::{BoosterKind, Device, MultiStrategy, TrainingParams, TreeMethod};
use crate::data::ghist::GHistIndex;
use crate::error::{HessboostError, Result};
use crate::model::BoostedModel;
use crate::objective::{GradPair, Loss, SplitGradient};
use crate::rng::Rng;
use crate::training::sampling::gradient_based_sample;
use crate::training::sglb::LeafRenewal;
use crate::tree::RegTree;
use crate::tree::builder::{LeafRows, MultiTreeBuilder, VectorGradients};
use crate::tree::constraints::MonotoneConstraints;

/// Whether training grows vector-leaf trees: `multi_output_tree` with more
/// than one output. A single output always gets scalar trees, as in XGBoost.
pub(super) fn vector_leaf(params: &TrainingParams, n_outputs: usize) -> bool {
    params.multi_strategy == MultiStrategy::MultiOutputTree
        && params.booster != BoosterKind::GbLinear
        && n_outputs > 1
}

/// Configuration checks for `multi_output_tree`: like XGBoost, vector-leaf
/// trees are built by the histogram method only, and their builder keeps
/// its own histogram loop, which no GPU backend accelerates.
pub(super) fn validate(params: &TrainingParams, n_outputs: usize) -> Result<()> {
    if params.multi_strategy == MultiStrategy::MultiOutputTree
        && params.booster != BoosterKind::GbLinear
        && !matches!(params.tree_method, TreeMethod::Hist | TreeMethod::Auto)
    {
        return Err(HessboostError::invalid_param(
            "multi_strategy",
            "`multi_output_tree` requires `tree_method=hist` (or `auto`)",
        ));
    }
    if vector_leaf(params, n_outputs) && params.device != Device::Cpu {
        return Err(HessboostError::invalid_param(
            "device",
            format!(
                "`{}` does not support `multi_strategy = multi_output_tree` \
                 (the vector-leaf builder has its own histogram loop)",
                params.device
            ),
        ));
    }
    Ok(())
}

/// Reduced split gradients are defined for vector-leaf trees only: refuse an
/// objective that supplies them to any other booster or strategy.
pub(crate) fn reject_split_gradient(
    objective: &dyn Loss,
    round: usize,
    gpair: &[GradPair],
) -> Result<()> {
    if objective.split_gradient(round, gpair).is_some() {
        return Err(HessboostError::invalid_param(
            "objective",
            "reduced split gradients require `multi_strategy=multi_output_tree` \
             with more than one output",
        ));
    }
    Ok(())
}

/// The objective's split gradients for this round, shape-checked.
fn split_gradient(
    objective: &dyn Loss,
    params: &TrainingParams,
    round: usize,
    gpair: &[GradPair],
    n_rows: usize,
) -> Result<Option<SplitGradient>> {
    let Some(split) = objective.split_gradient(round, gpair) else {
        return Ok(None);
    };
    if split.n_targets == 0 || Some(split.gpair.len()) != n_rows.checked_mul(split.n_targets) {
        return Err(HessboostError::dimension_mismatch(
            "split gradient length (n_rows * split n_targets)",
            n_rows.saturating_mul(split.n_targets.max(1)),
            split.gpair.len(),
        ));
    }
    if MonotoneConstraints::from_params(&params.monotone_constraints).is_active() {
        return Err(HessboostError::invalid_param(
            "monotone_constraints",
            "monotone constraints are not supported with reduced split gradients",
        ));
    }
    Ok(Some(split))
}

/// What every vector-leaf round reads: the run's shared inputs and the
/// training matrix's gradient index.
pub(super) struct VectorRound<'a> {
    pub(super) run: TrainContext<'a>,
    pub(super) ghist: &'a GHistIndex,
    /// Every training row, ascending: the rows of an unsampled round.
    pub(super) all_rows: &'a [u32],
}

/// One boosting iteration: grow `num_parallel_tree` vector-leaf trees from
/// the gradients at the current margins (DART: the ensemble minus this
/// iteration's dropout set), add them to the model, and bring the train and
/// eval margin caches up to date. `iteration` is the model's absolute
/// iteration index (continued training counts on), which seeds the RNG.
/// `noisy` holds SGLB's structure gradients (unused otherwise).
pub(super) fn boost_round(
    ctx: &VectorRound,
    model: &mut BoostedModel,
    iteration: usize,
    margins: &mut MarginCaches,
    gpair: &mut [GradPair],
    noisy: &mut Vec<GradPair>,
) -> Result<()> {
    let params = ctx.run.params;
    let n = ctx.run.dtrain.n_rows();
    let n_out = model.n_outputs();
    let (mut rng, dropped) = round_gradients(&ctx.run, model, iteration, &margins.train, gpair);
    let split = split_gradient(ctx.run.objective, params, iteration, gpair, n)?;
    let weight = dart_new_tree_weight(dropped.as_ref(), params);
    // SGLB: the structure grows on the split gradients (or the gradients)
    // plus noise; the builder's leaves then come from them too, and the
    // re-estimation replaces them.
    let structure = ctx.run.langevin.map(|langevin| {
        let searched = split.as_ref().map_or(&gpair[..], |s| &s.gpair[..]);
        langevin.structure_gradients(searched, iteration, noisy)
    });
    let grads = IterationGradients {
        gpair,
        split: split.as_ref(),
        structure,
        n_out,
        iteration,
    };
    // The row samples, all drawn before the trees: one per parallel tree
    // under uniform sampling, else one all-rows subset they share.
    let row_subsets = iteration_row_subsets(params, false, ctx.run.rows, ctx.all_rows, &mut rng);
    for p in 0..params.num_parallel_tree {
        let rows = row_subsets.rows(p);
        let (tree, leaf_rows) = fit_tree(ctx, &grads, &mut rng, rows)?;
        // A dropout round's gradients come from the ensemble, not the margin
        // caches, which `finish_dart` recomputes.
        if dropped.is_none() {
            // Leaf row lists identify every training row's leaf when all
            // rows took part in growing the tree.
            let captured =
                (rows.len() == n && !gradient_sampling(params)).then_some(leaf_rows.as_slice());
            margins.add_tree(&tree, TreeOutput::Vector, captured);
        }
        model.push_tree_weighted(tree, weight);
    }
    if let Some(dropped) = &dropped {
        finish_dart(model, params, dropped, margins);
    }
    Ok(())
}

/// The gradients of one vector-leaf iteration: every output's gradients
/// (`[row][n_out]`, also the leaf values' unless the objective splits on
/// reduced ones), the objective's split gradients, and SGLB's noisy
/// structure gradients, which replace the split gradients in the search.
struct IterationGradients<'a> {
    gpair: &'a [GradPair],
    split: Option<&'a SplitGradient>,
    structure: Option<&'a [GradPair]>,
    n_out: usize,
    iteration: usize,
}

/// Grow one vector-leaf tree: gradient-based row sampling (on the split
/// gradients, replayed on the value gradients), the tree's column sampler,
/// the build, and `eta / num_parallel_tree` shrinkage of every leaf vector.
/// Under SGLB the structure grows on the noisy gradients and the leaf
/// vectors are re-estimated from `gpair` (validation guarantees
/// `num_parallel_tree = 1` and no gradient-based sampling then).
fn fit_tree(
    ctx: &VectorRound,
    grads: &IterationGradients,
    rng: &mut Rng,
    rows: &[u32],
) -> Result<(RegTree, Vec<LeafRows>)> {
    let params = ctx.run.params;
    let IterationGradients {
        gpair,
        split,
        n_out,
        ..
    } = *grads;
    let (split_gpair, n_split) = split.map_or((gpair, n_out), |s| (&s.gpair[..], s.n_targets));
    let split_gpair = grads.structure.unwrap_or(split_gpair);
    let sampled = if gradient_sampling(params) {
        gradient_based_sample(split_gpair, n_split, params.subsample, rng)?
    } else {
        None
    };
    let sampled_value = match (&sampled, split) {
        (Some(sample), Some(_)) => Some(sample.apply(gpair, n_out)),
        _ => None,
    };
    let (split_gpair, rows) = match &sampled {
        Some(sample) => (sample.gpair.as_slice(), sample.rows.as_slice()),
        None => (split_gpair, rows),
    };
    let value = split.map(|_| sampled_value.as_deref().unwrap_or(gpair));
    let mut sampler = make_column_sampler(ctx.run.dtrain, params, rng);
    let grad = VectorGradients {
        split: split_gpair,
        n_split,
        value,
        n_outputs: n_out,
    };
    let (mut tree, leaf_rows) =
        MultiTreeBuilder::new(params).build(ctx.ghist, &grad, rows, &mut sampler);
    if let Some(langevin) = ctx.run.langevin {
        let at = LeafRenewal {
            data: ctx.run.dtrain,
            gpair,
            n_out,
            rows,
            leaf_rows: &leaf_rows,
            iteration: grads.iteration,
            tree: 0,
        };
        langevin.renew_leaves(&mut tree, TreeOutput::Vector, &at);
    }
    tree.scale_leaves(tree_eta(params));
    Ok((tree, leaf_rows))
}
