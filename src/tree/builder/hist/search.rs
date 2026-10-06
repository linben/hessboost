//! The histogram builder's split search: every sampled feature's numeric
//! scans (a wide node's in parallel chunks) and categorical sweeps, merged in
//! feature order with XGBoost's tie rule.

use super::{HistTreeBuilder, NodeCtx};
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::tree::builder::categorical::sweep_categorical;
use crate::tree::builder::shared::{InteractionState, permits, rayon_available, xgb_node_gain};
use crate::tree::builder::split::{
    NumericInput, NumericScan, SplitScorer, for_each_numeric_split, scan_numeric_pair,
    with_scan_scratch,
};
use crate::tree::builder::{BestSplit, need_replace, xgb_update};
use crate::tree::gain::GradStats;
use crate::tree::reuse::CategoricalPenalty;
use rayon::prelude::*;
use std::borrow::Cow;

/// Split candidates (feature bins) at which a node's numeric scans run in
/// parallel chunks of about [`SCAN_TASK_BINS`] candidates each.
const PARALLEL_SCAN_BINS: usize = 4096;
const SCAN_TASK_BINS: usize = 2048;

impl HistTreeBuilder<'_> {
    /// Find the best split for `node` from its histogram, enumerating each
    /// sampled feature's bins as XGBoost's histogram evaluator does
    /// ([`for_each_numeric_split`]). Candidates are scored and compared with
    /// XGBoost's `f32` arithmetic and tie rule, so near-equal gains resolve
    /// the same way. Monotone bounds are honored through the bounded child
    /// weights. With LightGBM split options enabled the search is delegated
    /// to [`SplitOptions::evaluate`](crate::tree::builder::lightgbm::SplitOptions::evaluate).
    pub(super) fn evaluate(
        &self,
        ghist: &GHistIndex,
        hist: &[GradStats],
        feature_subset: &[u32],
        allowed: Option<&InteractionState>,
        node: NodeCtx,
    ) -> BestSplit {
        let feature_subset = permitted(feature_subset, allowed);
        if let Some(options) = &self.options {
            return options.evaluate(
                ghist,
                hist,
                &feature_subset,
                &self.config.cons,
                &self.config.reg,
                node,
            );
        }
        let cuts = ghist.cuts();
        // A dense index has no missing entries: every feature's bins sum to
        // `total`, so the missing direction is never distinct and the
        // per-feature sums need not be computed.
        let dense = ghist.dense_stride().is_some();
        let total = node.stats;
        let node_scorer = self.node_scorer(node);
        // [`scan_numeric_splits`] of every plain numeric feature of `chunk`
        // (by position; `None` for the others), two at a time so their
        // prefix-sum chains overlap.
        let scan_chunk = |chunk: &[u32]| -> Vec<Option<NumericScan>> {
            let input =
                |f: u32| numeric_input(cuts, hist, f, total, dense, self.scorer(node_scorer, f));
            let mut out: Vec<Option<NumericScan>> = chunk.iter().map(|_| None).collect();
            with_scan_scratch(|[sa, sb]| {
                let mut pending = None;
                for (i, &f) in chunk.iter().enumerate() {
                    if !plain_numeric(cuts, f) {
                        continue;
                    }
                    match pending.take() {
                        None => pending = Some(i),
                        Some(j) => {
                            let [x, y] = scan_numeric_pair(&input(chunk[j]), &input(f), [sa, sb]);
                            (out[j], out[i]) = (Some(x), Some(y));
                        }
                    }
                }
                if let Some(j) = pending {
                    out[j] = Some(input(chunk[j]).scan(sa));
                }
            });
            out
        };
        // The scans of plain numeric features do not depend on the
        // incumbent, so they are computed up front (a wide search in
        // parallel); they are then merged in feature order.
        let scans = if self.reuse.is_some() {
            None
        } else {
            Some(
                Self::parallel_scans(cuts, &feature_subset, scan_chunk)
                    .unwrap_or_else(|| scan_chunk(&feature_subset)),
            )
        };
        // Never `None` given the histogram.
        self.merge(ghist, Some(hist), &feature_subset, node, scans)
            .unwrap_or_else(BestSplit::none)
    }

    /// The node's scorer (no monotone direction yet).
    pub(super) fn node_scorer(&self, node: NodeCtx) -> SplitScorer<'_> {
        SplitScorer {
            reg: &self.config.reg,
            root_gain: xgb_node_gain(node.stats, &self.config.reg, node.bounds),
            bounds: node.bounds,
            dir: 0,
        }
    }

    /// `node_scorer` with feature `f`'s monotone direction.
    pub(super) fn scorer<'s>(&self, node_scorer: SplitScorer<'s>, f: u32) -> SplitScorer<'s> {
        SplitScorer {
            dir: self.config.cons.dir(f as usize),
            ..node_scorer
        }
    }

    /// The best split of `feature_subset` (already restricted to the
    /// permitted features) in feature order with XGBoost's tie rule, from
    /// `scans` (each plain numeric feature's [`scan_numeric_splits`] by
    /// position, `None` to scan here) and the node's histogram, which is
    /// read for categorical features, reuse penalties, NaN scans, and
    /// features without a scan. `None` when one of those needs it and
    /// `hist` is `None`.
    pub(super) fn merge(
        &self,
        ghist: &GHistIndex,
        hist: Option<&[GradStats]>,
        feature_subset: &[u32],
        node: NodeCtx,
        mut scans: Option<Vec<Option<NumericScan>>>,
    ) -> Option<BestSplit> {
        let cuts = ghist.cuts();
        let dense = ghist.dense_stride().is_some();
        let total = node.stats;
        let node_scorer = self.node_scorer(node);
        let mut best = BestSplit::none();
        for (i, &f) in feature_subset.iter().enumerate() {
            let (fs, fe) = cuts.feature_bins(f as usize);
            let scorer = self.scorer(node_scorer, f);

            if cuts.is_categorical(f as usize) {
                let hist = hist?;
                // Every category bin, empty ones included, as XGBoost
                // enumerates them (a lone category can still split present
                // from missing values).
                let cats: Vec<(u32, GradStats)> = (fs..fe)
                    .map(|i| (cuts.cut_value(i) as u32, hist[i]))
                    .collect();
                sweep_categorical(
                    &mut best,
                    &cats,
                    total,
                    &scorer,
                    f,
                    self.reuse.as_ref().map(|r| r as &dyn CategoricalPenalty),
                );
                continue;
            }
            if fe <= fs + 1 {
                continue; // degenerate feature, no interior boundary
            }

            if let Some(reuse) = &self.reuse {
                for_each_numeric_split(&hist?[fs..fe], fs, total, dense, |pos, children| {
                    let Some(mut score) = scorer.loss_chg(children.left, children.right) else {
                        return;
                    };
                    score.loss_chg -= reuse.bin_penalty(f, pos.bin());
                    xgb_update(&mut best, f, pos, children, score);
                });
                continue;
            }
            let scanned = if let Some(scan) = scans.as_mut().and_then(|scans| scans[i].take()) {
                scan
            } else {
                let input = numeric_input(cuts, hist?, f, total, dense, scorer);
                with_scan_scratch(|[s, _]| input.scan(s))
            };
            match scanned {
                NumericScan::Empty => {}
                NumericScan::Best {
                    loss_chg,
                    pos,
                    children,
                } => {
                    if need_replace(best.loss_chg as f32, best.feature, loss_chg, f)
                        && let Some(score) = scorer.loss_chg(children.left, children.right)
                    {
                        xgb_update(&mut best, f, pos, children, score);
                    }
                }
                NumericScan::Nan => {
                    for_each_numeric_split(&hist?[fs..fe], fs, total, dense, |pos, children| {
                        if let Some(score) = scorer.loss_chg(children.left, children.right) {
                            xgb_update(&mut best, f, pos, children, score);
                        }
                    });
                }
            }
        }
        Some(best)
    }

    /// `scan_chunk` over parallel chunks of `feature_subset`, concatenated,
    /// when the subset holds enough candidates to pay for the tasks. `None`
    /// otherwise.
    fn parallel_scans(
        cuts: &HistCuts,
        feature_subset: &[u32],
        scan_chunk: impl Fn(&[u32]) -> Vec<Option<NumericScan>> + Sync,
    ) -> Option<Vec<Option<NumericScan>>> {
        if !rayon_available() {
            return None;
        }
        let bins = |f: u32| {
            let (fs, fe) = cuts.feature_bins(f as usize);
            fe - fs
        };
        let candidates: usize = feature_subset.iter().map(|&f| bins(f)).sum();
        if candidates < PARALLEL_SCAN_BINS {
            return None;
        }
        let per_task = (SCAN_TASK_BINS * feature_subset.len()).div_ceil(candidates);
        Some(
            feature_subset
                .par_chunks(per_task.max(1))
                .flat_map_iter(&scan_chunk)
                .collect(),
        )
    }
}

/// `feature_subset` restricted to the features the interaction state
/// `allowed` permits (a sorted set; `None`: every feature).
pub(super) fn permitted<'a>(
    feature_subset: &'a [u32],
    allowed: Option<&InteractionState>,
) -> Cow<'a, [u32]> {
    match allowed {
        Some(_) => feature_subset
            .iter()
            .copied()
            .filter(|&f| permits(allowed, f))
            .collect(),
        None => Cow::Borrowed(feature_subset),
    }
}

/// Whether feature `f` is numeric with an interior boundary: the features
/// [`scan_numeric_splits`](crate::tree::builder::split::scan_numeric_splits)
/// scans.
pub(super) fn plain_numeric(cuts: &HistCuts, f: u32) -> bool {
    let (fs, fe) = cuts.feature_bins(f as usize);
    !cuts.is_categorical(f as usize) && fe > fs + 1
}

/// Feature `f`'s numeric scan input from the node histogram `hist`.
fn numeric_input<'a>(
    cuts: &HistCuts,
    hist: &'a [GradStats],
    f: u32,
    total: GradStats,
    dense: bool,
    scorer: SplitScorer<'a>,
) -> NumericInput<'a> {
    let (fs, fe) = cuts.feature_bins(f as usize);
    NumericInput {
        bins: &hist[fs..fe],
        first: fs,
        total,
        dense,
        scorer,
    }
}
