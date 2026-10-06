//! Histogram-based tree construction (XGBoost's `tree_method=hist`).
//!
//! Features are pre-binned once ([`GHistIndex`]). Growing a node reduces to
//! scanning its per-bin gradient histogram. Sibling histograms are obtained by
//! subtraction (`sibling = parent − smaller_child`), so only the smaller child
//! is ever built directly. Supports `depthwise` and `lossguide` growth, and
//! hands `symmetric` growth to the level-wise oblivious builder.

mod device;
mod search;

use super::lightgbm::{SplitOptions, finalize_smoothed_leaves};
use super::partition::{child_histograms, partition_rows};
use super::shared::{
    BuilderConfig, InteractionState, LeafRows, finalize_leaf_values, rayon_available, sum_rows,
    xgb_calc_weight,
};
use super::{BELOW_ALL_VALUES, BestSplit, SplitLocation, limit_or_unbounded};
use crate::config::{GrowPolicy, TrainingParams};
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::objective::GradPair;
use crate::tree::constraints::Bounds;
use crate::tree::gain::GradStats;
use crate::tree::hist::quantized::QuantNode;
use crate::tree::hist::{CpuBackend, HistSlot, Histogram, HistogramBackend, Segment, zeroed};
use crate::tree::regtree::RegTree;
use crate::tree::reuse::{HistReuse, ReuseSet};
use crate::tree::sampler::{ColumnSampler, FeatureSet};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Queued loss-guided nodes whose children are built together, in parallel
/// (see [`HistTreeBuilder::speculative_features`]).
const SPECULATE_NODES: usize = 8;

/// Nodes with at least this many rows evaluate their two children's splits
/// concurrently. Smaller nodes appear in frontiers wide enough to keep the
/// pool busy, and their evaluation is too short to be worth a fork.
const PARALLEL_EVALUATE_ROWS: usize = 4096;

/// Combined level rows at which depthwise and symmetric growth build a
/// level's child histograms concurrently. Below this, the fork costs more
/// than the scan.
pub(super) const PARALLEL_FRONTIER_ROWS: usize = 4096;

/// The node a split search runs for.
#[derive(Debug, Clone, Copy)]
pub(super) struct NodeCtx {
    /// Node id in the tree being grown; seeds the node's `extra_trees` draws.
    pub(super) id: usize,
    /// The node's gradient statistics, including missing values.
    pub(super) stats: GradStats,
    /// The node's monotone weight bounds.
    pub(super) bounds: Bounds,
    /// Training rows in the node (`n` of path smoothing).
    pub(super) rows: usize,
    /// The node's own output, which path smoothing pulls its children toward.
    pub(super) output: f64,
    /// Seed of the tree being grown ([`crate::tree::sampler::ColumnSampler::seed`]).
    pub(super) tree_seed: u64,
}

/// A node awaiting or undergoing expansion.
struct NodeEntry {
    nid: usize,
    depth: usize,
    /// The node's rows on the host (empty when `seg` holds them).
    rows: Vec<u32>,
    /// The node's rows on the device, under device-resident growth.
    seg: Option<Segment>,
    hist: Histogram,
    best: BestSplit,
    bounds: Bounds,
    /// Features permitted for splits under this node given the interaction
    /// constraints and the split features on the path from the root. `None`
    /// means "all features allowed" (the root, and the inactive case).
    allowed: Option<InteractionState>,
    /// The tree's seed, handed to every node's split search.
    tree_seed: u64,
    /// Quantized histogram (`use_quantized_grad`); `hist` then holds its
    /// dequantized copy for split evaluation.
    quant: Option<QuantNode>,
    /// The node's histogram in a row engine's slot under resident growth
    /// (`hist` is then empty).
    slot: Option<HistSlot>,
}

/// Tree expansion and sampling happen in node order, so the expensive row and
/// histogram work can then run independently for every split at a depth.
struct PendingSplit {
    entry: NodeEntry,
    left_id: usize,
    right_id: usize,
    left_bounds: Bounds,
    right_bounds: Bounds,
    left_features: FeatureSet,
    right_features: FeatureSet,
    /// Depthwise children at the depth limit: they stay leaves, so they need
    /// no histograms or split searches, only their rows (kept only when leaf
    /// rows are captured; otherwise the split is never built).
    terminal: bool,
}

/// The CPU backend every builder defaults to.
pub(super) static CPU_BACKEND: CpuBackend = CpuBackend;
// Ordering for the loss-guided priority queue (max-heap on loss change).
impl PartialEq for NodeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.best.loss_chg == other.best.loss_chg
    }
}
impl Eq for NodeEntry {}
impl PartialOrd for NodeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NodeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.best.loss_chg.total_cmp(&other.best.loss_chg)
    }
}

/// Histogram tree builder.
pub struct HistTreeBuilder<'a> {
    config: BuilderConfig<'a>,
    /// Histogram construction backend (the CPU's, or a GPU's when
    /// `device = metal`).
    backend: &'a dyn HistogramBackend,
    /// LightGBM `extra_trees` / `path_smooth`; `None` keeps XGBoost's search.
    options: Option<SplitOptions>,
    /// Opt-in reuse penalties (`toad_penalty_*`), projected onto the bins of
    /// the index this builder grows on. `None` on the default path.
    reuse: Option<HistReuse>,
    /// Stream of the stochastic gradient rounding (`use_quantized_grad`).
    rounding_seed: u64,
}

impl<'a> HistTreeBuilder<'a> {
    /// Create a builder bound to a training configuration.
    pub fn new(params: &'a TrainingParams) -> Self {
        HistTreeBuilder {
            config: BuilderConfig::new(params),
            backend: &CPU_BACKEND,
            options: SplitOptions::from_params(params),
            reuse: None,
            rounding_seed: 0,
        }
    }

    /// Penalize candidates by the reuse penalties of `set` (the ensemble's
    /// used features and thresholds; `None` keeps the default gain). `cuts`
    /// must be the cuts of every index this builder grows on. Splits the
    /// builder commits extend its own copy, so later nodes (and later trees
    /// grown by this builder) reuse them for free.
    #[must_use]
    pub(crate) fn with_reuse(mut self, set: Option<&ReuseSet>, cuts: &HistCuts) -> Self {
        self.reuse = set.map(|set| HistReuse::new(set, cuts, BELOW_ALL_VALUES));
        self
    }
    /// Seed the stochastic rounding of quantized training
    /// (`use_quantized_grad`). The trainer passes a distinct seed per round
    /// and output so rounding noise is independent across trees.
    #[must_use]
    pub(crate) fn with_rounding_seed(mut self, seed: u64) -> Self {
        self.rounding_seed = seed;
        self
    }

    /// Use a specific histogram backend (the Metal GPU's, when
    /// `device = metal`). Held by reference: the trainer owns it for the
    /// whole run.
    #[must_use]
    pub(crate) fn with_backend(mut self, backend: &'a dyn HistogramBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Grow one tree from the binned dataset.
    ///
    /// * `ghist`: the binned dataset (built once, reused across rounds).
    /// * `gpair`: per-row gradient/Hessian (length = dataset rows).
    /// * `row_subset`: sampled rows for this tree.
    /// * `sampler`: per-tree column sampler. A fresh subset is drawn per node.
    pub fn build(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
    ) -> RegTree {
        self.build_inner(ghist, gpair, row_subset, sampler, false).0
    }

    /// Keep the final row partitions so training can update margins without
    /// traversing the tree again. Used without row sampling.
    pub(crate) fn build_with_leaf_rows(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
    ) -> (RegTree, Vec<LeafRows>) {
        self.build_inner(ghist, gpair, row_subset, sampler, true)
    }

    fn build_inner(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
        capture_rows: bool,
    ) -> (RegTree, Vec<LeafRows>) {
        // Stage the gradients once per tree (a GPU backend uploads them
        // here); every node build below reads the same slice.
        self.backend.prepare(ghist, gpair);

        if self.config.params.grow_policy == GrowPolicy::Symmetric {
            return super::oblivious::SymmetricTreeBuilder::new(&self.config, self.backend).build(
                ghist,
                gpair,
                row_subset,
                sampler,
                capture_rows,
            );
        }
        debug_assert!(
            self.reuse
                .as_ref()
                .is_none_or(|r| r.n_bins() == ghist.total_bins())
        );
        // A device that keeps the rows grows the tree there. It may fail
        // part way (a device error); the tree is then regrown here from the
        // sampler's state at the start, so the result is unchanged.
        if let Some(engine) = self.device_engine() {
            let start = sampler.clone();
            let report = if capture_rows {
                device::LeafReport::Rows
            } else {
                device::LeafReport::Nothing
            };
            if let Some((tree, leaf_rows, _)) =
                self.build_on_device(engine, ghist, Some(gpair), row_subset, sampler, report)
            {
                return (tree, leaf_rows);
            }
            *sampler = start;
        }
        // Leaf renewal recomputes leaf values from full-precision sums, which
        // needs every leaf's rows.
        let renew = self.config.params.quantized.is_some_and(|q| q.renew_leaf());
        let (root, root_stats) = self.root(ghist, gpair, row_subset, sampler);
        let mut tree = RegTree::with_root(root_stats.hess as f32);
        let mut store = NodeStore::new(root_stats, capture_rows || renew);

        match self.config.params.grow_policy {
            GrowPolicy::DepthWise => {
                self.grow_depthwise(&mut tree, &mut store, ghist, gpair, sampler, root);
            }
            GrowPolicy::LossGuide => {
                self.grow_lossguide(&mut tree, &mut store, ghist, gpair, sampler, root);
            }
            GrowPolicy::Symmetric => unreachable!("symmetric trees return above"),
        }

        if renew && let Some(leaves) = &store.leaf_rows {
            for leaf in leaves {
                store.stats[leaf.node] = sum_rows(gpair, &leaf.rows);
            }
        }
        self.finish_tree(&mut tree, &store, root_stats);
        let leaf_rows = if capture_rows {
            store.leaf_rows.unwrap_or_default()
        } else {
            Vec::new()
        };
        (tree, leaf_rows)
    }

    /// Finalize leaf weights (respecting each leaf's monotone bounds).
    /// Path-smoothed leaves already hold the outputs their splits chose.
    fn finish_tree(&self, tree: &mut RegTree, store: &NodeStore, root_stats: GradStats) {
        match &self.options {
            Some(options) if options.smoothing() => {
                finalize_smoothed_leaves(tree, root_stats, &self.config.reg);
            }
            _ => finalize_leaf_values(tree, &store.stats, &store.bounds, &self.config.reg),
        }
    }

    /// The root node (its histogram, statistics, and best split) of a tree
    /// over `row_subset`, drawing the root's column sample.
    fn root(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
    ) -> (NodeEntry, GradStats) {
        let total_bins = ghist.total_bins();
        let (root_stats, root_hist, root_quant) =
            if let Some(quantized) = self.config.params.quantized {
                let (quant, stats, hist) =
                    QuantNode::root(ghist, gpair, row_subset, quantized, self.rounding_seed);
                (stats, hist, Some(quant))
            } else {
                // The root sum is a sequential pass; it runs beside the
                // (parallel) root histogram instead of before it.
                let build_hist = || {
                    let mut root_hist = zeroed(total_bins);
                    self.backend.build(ghist, row_subset, gpair, &mut root_hist);
                    root_hist
                };
                let (root_stats, root_hist) = if rayon_available() {
                    rayon::join(|| sum_rows(gpair, row_subset), build_hist)
                } else {
                    (sum_rows(gpair, row_subset), build_hist())
                };
                (root_stats, root_hist, None)
            };
        let root = self.root_entry(ghist, sampler, root_stats, root_hist, row_subset.len());
        let root = NodeEntry {
            rows: row_subset.to_vec(),
            quant: root_quant,
            ..root
        };
        (root, root_stats)
    }

    /// The root entry of `rows` rows with statistics `root_stats` and
    /// histogram `root_hist`, drawing the root's column sample and finding
    /// its best split. Its rows are left empty for the caller.
    fn root_entry(
        &self,
        ghist: &GHistIndex,
        sampler: &mut ColumnSampler,
        root_stats: GradStats,
        root_hist: Histogram,
        rows: usize,
    ) -> NodeEntry {
        let (root_feats, root_ctx) = self.root_context(sampler, root_stats, rows);
        let best = self.evaluate(ghist, &root_hist, &root_feats, None, root_ctx);
        NodeEntry {
            hist: root_hist,
            ..Self::root_node(root_ctx.tree_seed, best)
        }
    }

    /// The root's sampled features and split-search context.
    fn root_context(
        &self,
        sampler: &mut ColumnSampler,
        root_stats: GradStats,
        rows: usize,
    ) -> (FeatureSet, NodeCtx) {
        // Per-node column sampling (bylevel ∘ bynode) draws a fresh subset here.
        let root_feats = sampler.sample(0);
        let tree_seed = sampler.seed();
        let root_ctx = NodeCtx {
            id: 0,
            stats: root_stats,
            bounds: Bounds::default(),
            rows,
            output: xgb_calc_weight(root_stats, &self.config.reg),
            tree_seed,
        };
        (root_feats, root_ctx)
    }

    /// The root's entry with split `best`, no rows and no histogram.
    fn root_node(tree_seed: u64, best: BestSplit) -> NodeEntry {
        NodeEntry {
            nid: 0,
            depth: 0,
            rows: Vec::new(),
            seg: None,
            hist: Vec::new(),
            best,
            bounds: Bounds::default(),
            allowed: None,
            tree_seed,
            quant: None,
            slot: None,
        }
    }

    fn grow_depthwise(
        &self,
        tree: &mut RegTree,
        store: &mut NodeStore,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        sampler: &mut ColumnSampler,
        root: NodeEntry,
    ) {
        let limit = limit_or_unbounded(self.config.params.max_depth);
        let mut frontier = vec![root];
        let mut depth = 0;
        while depth < limit && !frontier.is_empty() {
            let parallel = frontier.len() > 1
                && frontier.iter().map(|entry| entry.rows.len()).sum::<usize>()
                    >= PARALLEL_FRONTIER_ROWS
                && rayon_available();
            let mut pending = Vec::with_capacity(frontier.len());
            for entry in frontier.drain(..) {
                if self.valid(&entry.best) {
                    if let Some(split) =
                        self.prepare_split(tree, store, ghist.cuts(), sampler, entry)
                    {
                        pending.push(split);
                    }
                } else {
                    store.record_leaf(entry);
                }
            }
            let build = |split| self.build_children(ghist, gpair, split);
            let children: Vec<_> = if parallel {
                pending.into_par_iter().map(build).collect()
            } else {
                pending.into_iter().map(build).collect()
            };
            frontier = children
                .into_iter()
                .flat_map(|(left, right)| [left, right])
                .collect();
            depth += 1;
        }
        for entry in frontier {
            store.record_leaf(entry);
        }
    }

    fn grow_lossguide(
        &self,
        tree: &mut RegTree,
        store: &mut NodeStore,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        sampler: &mut ColumnSampler,
        root: NodeEntry,
    ) {
        let limit = limit_or_unbounded(self.config.params.max_depth);
        let max_leaves = limit_or_unbounded(self.config.params.max_leaves);
        let expandable = |entry: &NodeEntry| entry.depth < limit && self.valid(&entry.best);
        let speculative = self.speculative_features(sampler);
        // Children built ahead of their parent's turn, by parent node id.
        let mut ready: HashMap<usize, (NodeEntry, NodeEntry)> = HashMap::new();
        let mut heap = BinaryHeap::new();
        heap.push(root);
        let mut n_leaves = 1usize;
        while let Some(entry) = heap.pop() {
            if n_leaves >= max_leaves {
                store.record_leaf(entry);
                break;
            }
            if !expandable(&entry) {
                store.record_leaf(entry);
                continue; // permanent leaf
            }
            let children = if let Some(features) = &speculative {
                if !ready.contains_key(&entry.nid) {
                    // Build this node's children together with those of the
                    // queue's next best candidates, at most as many as can
                    // still be expanded.
                    let budget = (max_leaves - n_leaves).min(SPECULATE_NODES);
                    let queued = heap
                        .iter()
                        .filter(|e| expandable(e) && !ready.contains_key(&e.nid));
                    let batch = speculation_batch(&entry, queued, budget);
                    let built: Vec<_> = batch
                        .par_iter()
                        .map(|e| (e.nid, self.speculate_children(ghist, gpair, e, features)))
                        .collect();
                    ready.extend(built);
                }
                let built = ready.remove(&entry.nid);
                // The expansion itself (node ids, stored statistics, sampler
                // draws) stays in queue order.
                self.prepare_split(tree, store, ghist.cuts(), sampler, entry)
                    .map(|split| match built {
                        Some((mut left, mut right)) => {
                            left.nid = split.left_id;
                            right.nid = split.right_id;
                            (left, right)
                        }
                        None => self.build_children(ghist, gpair, split),
                    })
            } else {
                self.prepare_split(tree, store, ghist.cuts(), sampler, entry)
                    .map(|split| self.build_children(ghist, gpair, split))
            };
            n_leaves += 1; // one leaf became two
            if let Some((l, r)) = children {
                heap.push(l);
                heap.push(r);
            }
        }
        for entry in heap {
            store.record_leaf(entry);
        }
    }

    /// The features every node samples, when loss-guided growth may build
    /// children ahead of their parent's turn: a node's children then depend
    /// only on the node, not on the expansion order. That needs a sampler
    /// without per-level or per-node draws, no reuse penalties (which each
    /// expansion extends), no LightGBM options (which key draws by node id),
    /// no quantized histograms, and more than one worker.
    fn speculative_features(&self, sampler: &ColumnSampler) -> Option<FeatureSet> {
        if self.reuse.is_some()
            || self.options.is_some()
            || self.config.params.quantized.is_some()
            || !rayon_available()
        {
            return None;
        }
        sampler.fixed_features()
    }

    /// [`Self::build_children`] of `entry` (whose split is valid) ahead of
    /// its turn, from copies of its rows and histogram. Node ids are
    /// placeholders the caller replaces once the expansion is due; the split
    /// search does not read them without LightGBM options.
    fn speculate_children(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        entry: &NodeEntry,
        features: &FeatureSet,
    ) -> (NodeEntry, NodeEntry) {
        let b = &entry.best;
        let (left_bounds, right_bounds) =
            b.child_bounds(entry.bounds, self.config.cons.dir(b.feature as usize));
        let split = PendingSplit {
            entry: NodeEntry {
                nid: entry.nid,
                depth: entry.depth,
                rows: entry.rows.clone(),
                seg: None,
                hist: entry.hist.clone(),
                best: entry.best.clone(),
                bounds: entry.bounds,
                allowed: entry.allowed.clone(),
                tree_seed: entry.tree_seed,
                quant: None,
                slot: None,
            },
            left_id: 0,
            right_id: 0,
            left_bounds,
            right_bounds,
            left_features: features.clone(),
            right_features: features.clone(),
            terminal: false,
        };
        self.build_children(ghist, gpair, split)
    }

    /// Whether a node's best split should be taken.
    fn valid(&self, best: &BestSplit) -> bool {
        best.valid(self.config.params.gamma, self.config.reg.min_child_weight)
    }

    /// Expand a node and draw child features in traversal order. Children at the
    /// depth limit need only stored statistics to finalize their leaf weights.
    fn prepare_split(
        &self,
        tree: &mut RegTree,
        store: &mut NodeStore,
        cuts: &HistCuts,
        sampler: &mut ColumnSampler,
        entry: NodeEntry,
    ) -> Option<PendingSplit> {
        let b = &entry.best;

        // Monotone child bounds derived from the (bounded) child weights.
        let dir = self.config.cons.dir(b.feature as usize);
        let (lb_bounds, rb_bounds) = b.child_bounds(entry.bounds, dir);

        let (left_id, right_id) = b.expand(tree, entry.nid, b.route().rule(cuts));
        if let Some(reuse) = &self.reuse {
            match &b.location {
                SplitLocation::Categories(categories) => {
                    reuse.commit_categorical(b.feature, categories);
                }
                SplitLocation::Numeric(pos) => reuse.commit_numeric(b.feature, pos.bin()),
            }
        }
        debug_assert_eq!(left_id, store.stats.len());
        store.push(b.left, lb_bounds);
        store.push(b.right, rb_bounds);

        let child_depth = entry.depth + 1;
        let terminal = self.config.params.grow_policy == GrowPolicy::DepthWise
            && child_depth >= limit_or_unbounded(self.config.params.max_depth);
        if terminal && store.leaf_rows.is_none() {
            // Preserve the draws for these two nodes, including when callers
            // reuse the sampler. Their rows, histograms and candidate splits
            // cannot affect this tree, and leaf weights use the stored stats.
            sampler.sample(child_depth);
            sampler.sample(child_depth);
            return None;
        }

        let left_features = sampler.sample(child_depth);
        let right_features = sampler.sample(child_depth);
        Some(PendingSplit {
            entry,
            left_id,
            right_id,
            left_bounds: lb_bounds,
            right_bounds: rb_bounds,
            left_features,
            right_features,
            terminal,
        })
    }

    fn build_children(
        &self,
        ghist: &GHistIndex,
        gpair: &[GradPair],
        mut split: PendingSplit,
    ) -> (NodeEntry, NodeEntry) {
        let parent_rows = std::mem::take(&mut split.entry.rows);
        let parent_hist = std::mem::take(&mut split.entry.hist);
        let parent_quant = split.entry.quant.take();
        let (left_rows, right_rows) = partition_rows(ghist, &parent_rows, split.entry.best.route());
        drop(parent_rows);

        let (mut left_quant, mut right_quant) = (None, None);
        let (left_hist, right_hist) = if split.terminal {
            (Vec::new(), Vec::new())
        } else if let Some(quant) = parent_quant {
            let ((lq, lh), (rq, rh)) = quant.children(ghist, &left_rows, &right_rows, parent_hist);
            (left_quant, right_quant) = (Some(lq), Some(rq));
            (lh, rh)
        } else {
            child_histograms(
                self.backend,
                ghist,
                gpair,
                &left_rows,
                &right_rows,
                parent_hist,
            )
        };
        let child = |rows: Vec<u32>, hist, quant| Child {
            len: rows.len(),
            rows,
            seg: None,
            hist,
            quant,
        };
        self.finish_children(
            ghist,
            split,
            child(left_rows, left_hist, left_quant),
            child(right_rows, right_hist, right_quant),
            None,
        )
    }

    /// The interaction state both children of `split` share and their
    /// split-search contexts (`left_len`/`right_len` rows).
    fn child_contexts(
        &self,
        split: &PendingSplit,
        left_len: usize,
        right_len: usize,
    ) -> (Option<InteractionState>, NodeCtx, NodeCtx) {
        let entry = &split.entry;
        let b = &entry.best;
        // Both children share the state derived from the complete updated path:
        // path features plus groups containing every feature on that path.
        let child_allowed = self.config.next_allowed(entry.allowed.as_ref(), b.feature);
        // Under path smoothing each child's output is the one its split
        // recorded; it is the parent output of the child's own children.
        let left_ctx = NodeCtx {
            id: split.left_id,
            stats: b.left,
            bounds: split.left_bounds,
            rows: left_len,
            output: b.w_left,
            tree_seed: entry.tree_seed,
        };
        let right_ctx = NodeCtx {
            id: split.right_id,
            stats: b.right,
            bounds: split.right_bounds,
            rows: right_len,
            output: b.w_right,
            tree_seed: entry.tree_seed,
        };
        (child_allowed, left_ctx, right_ctx)
    }

    /// Both children of `split` from their rows and histograms (empty for
    /// terminal splits): their interaction state and best splits, searched
    /// here unless `searched` holds them.
    fn finish_children(
        &self,
        ghist: &GHistIndex,
        split: PendingSplit,
        left: Child,
        right: Child,
        searched: Option<(BestSplit, BestSplit)>,
    ) -> (NodeEntry, NodeEntry) {
        let (child_allowed, left_ctx, right_ctx) = self.child_contexts(&split, left.len, right.len);
        let PendingSplit {
            entry,
            left_id,
            right_id,
            left_bounds: lb_bounds,
            right_bounds: rb_bounds,
            left_features,
            right_features,
            terminal,
        } = split;
        let NodeEntry {
            depth: parent_depth,
            tree_seed,
            ..
        } = entry;

        // The children's split searches are independent; near the root, where
        // the frontier holds too few nodes to occupy the pool, running them
        // side by side halves the serial evaluation time. Each search keeps
        // its sequential candidate order, so the chosen split is identical.
        let (left_best, right_best) = if terminal {
            (BestSplit::none(), BestSplit::none())
        } else if let Some(searched) = searched {
            searched
        } else {
            let allowed = child_allowed.as_ref();
            let eval_left = || self.evaluate(ghist, &left.hist, &left_features, allowed, left_ctx);
            let eval_right =
                || self.evaluate(ghist, &right.hist, &right_features, allowed, right_ctx);
            if left.len + right.len >= PARALLEL_EVALUATE_ROWS && rayon_available() {
                rayon::join(eval_left, eval_right)
            } else {
                (eval_left(), eval_right())
            }
        };

        let left = NodeEntry {
            nid: left_id,
            depth: parent_depth + 1,
            rows: left.rows,
            seg: left.seg,
            hist: left.hist,
            best: left_best,
            bounds: lb_bounds,
            allowed: child_allowed.clone(),
            tree_seed,
            quant: left.quant,
            slot: None,
        };
        let right = NodeEntry {
            nid: right_id,
            depth: parent_depth + 1,
            rows: right.rows,
            seg: right.seg,
            hist: right.hist,
            best: right_best,
            bounds: rb_bounds,
            allowed: child_allowed,
            tree_seed,
            quant: right.quant,
            slot: None,
        };
        (left, right)
    }
}

/// One child's rows (on the host, or a device segment) and histograms,
/// handed to [`HistTreeBuilder::finish_children`].
struct Child {
    len: usize,
    rows: Vec<u32>,
    seg: Option<Segment>,
    hist: Histogram,
    quant: Option<QuantNode>,
}

/// The nodes whose children loss-guided growth builds together: `entry` (the
/// node due now) and the best `budget - 1` of the other expandable `queued`
/// nodes, in queue order (the heap's order: largest loss change first).
fn speculation_batch<'e>(
    entry: &'e NodeEntry,
    queued: impl Iterator<Item = &'e NodeEntry>,
    budget: usize,
) -> Vec<&'e NodeEntry> {
    let mut queued: Vec<&NodeEntry> = queued.collect();
    queued.sort_by(|a, b| b.cmp(a));
    std::iter::once(entry)
        .chain(queued.into_iter().take(budget - 1))
        .collect()
}

/// Per-node statistics and monotone bounds, indexed by node id.
struct NodeStore {
    stats: Vec<GradStats>,
    bounds: Vec<Bounds>,
    leaf_rows: Option<Vec<LeafRows>>,
    /// Leaves whose rows are still on the device, in record order.
    device_leaves: Vec<(usize, Segment)>,
}

impl NodeStore {
    fn new(root_stats: GradStats, keep_rows: bool) -> Self {
        NodeStore {
            stats: vec![root_stats],
            bounds: vec![Bounds::default()],
            leaf_rows: keep_rows.then(Vec::new),
            device_leaves: Vec::new(),
        }
    }

    fn record_leaf(&mut self, entry: NodeEntry) {
        if let Some(leaves) = &mut self.leaf_rows {
            match entry.seg {
                Some(seg) => self.device_leaves.push((entry.nid, seg)),
                None => leaves.push(LeafRows {
                    node: entry.nid,
                    rows: entry.rows,
                }),
            }
        }
    }

    #[inline]
    fn push(&mut self, stats: GradStats, bounds: Bounds) {
        self.stats.push(stats);
        self.bounds.push(bounds);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        binned, gp, grow_exact, grow_hist, monotone_v_shape_data, non_decreasing,
    };
    use super::*;
    use crate::config::TrainingParams;
    use crate::data::DMatrix;
    use crate::tree::builder::all_rows;

    #[test]
    fn splits_on_separating_feature() {
        let x = vec![0.0f32, 0.0, 1.0, 1.0];
        let data = DMatrix::from_dense(&x, 4, 1).unwrap();
        let ghist = binned(&data, 256);
        let gpair = vec![gp(1.0, 1.0), gp(1.0, 1.0), gp(-1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .build()
            .unwrap();
        let tree = grow_hist(&params, &ghist, &gpair);
        assert_eq!(tree.num_nodes(), 3);
        assert!((tree.predict_row(&data, 0) - (-1.0)).abs() < 1e-6);
        assert!((tree.predict_row(&data, 2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn depth_limit_preserves_reused_column_sampler_state() {
        let n = 256;
        let features = 8;
        let x: Vec<f32> = (0..n * features)
            .map(|i| ((i / features * 17 + i % features * 29) % 97) as f32 / 97.0)
            .collect();
        let gradients: Vec<GradPair> = x
            .chunks_exact(features)
            .map(|row| gp(row.iter().sum::<f32>() - 4.0, 1.0))
            .collect();
        let data = DMatrix::from_dense(&x, n, features).unwrap();
        let ghist = binned(&data, 64);
        let rows = all_rows(n);

        for depth in [1, 2, 4] {
            let params = TrainingParams::builder().max_depth(depth).build().unwrap();
            let builder = HistTreeBuilder::new(&params);
            let new_sampler = || ColumnSampler::new(features, None, 1.0, 0.75, 0.75, 42);
            let mut sampler = new_sampler();
            let mut expected = new_sampler();
            for _ in 0..3 {
                let tree = builder.build(&ghist, &gradients, &rows, &mut sampler);
                assert!(tree.num_nodes() > 1);
                // Every created node consumes one draw at its depth, in node-id
                // order, including leaves whose histogram and split search are
                // skipped at the depth limit.
                let mut node_depth = vec![0usize; tree.num_nodes()];
                for (nid, node) in tree.nodes().iter().enumerate() {
                    expected.sample(node_depth[nid]);
                    if !node.is_leaf() {
                        node_depth[node.left as usize] = node_depth[nid] + 1;
                        node_depth[node.right as usize] = node_depth[nid] + 1;
                    }
                }
                for depth in 0..4 {
                    assert_eq!(sampler.sample(depth), expected.sample(depth));
                }
            }
        }
    }

    #[test]
    fn parallel_depthwise_preserves_tree_and_sampler() {
        use crate::config::Monotone;
        use crate::data::FeatureType;

        let n = 8192;
        let features = 8;
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for mode in ["dense", "missing", "categorical"] {
            let mut state = 123u64;
            let mut values = Vec::with_capacity(n * features);
            let mut gradients = Vec::with_capacity(n);
            for row in 0..n {
                let mut target = 0.0;
                for col in 0..features {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let mut value = (state >> 33) as f32 / (1u32 << 31) as f32;
                    if mode == "categorical" && col < 2 {
                        value = (value * 4.0).floor();
                    }
                    target += value * (col + 1) as f32;
                    if mode == "missing" && (row * 13 + col * 7) % 11 < 2 {
                        value = f32::NAN;
                    }
                    values.push(value);
                }
                gradients.push(gp(18.0 - target, 1.0));
            }
            let mut data = DMatrix::from_dense(&values, n, features).unwrap();
            if mode == "categorical" {
                let mut types = vec![FeatureType::Numerical; features];
                types[..2].fill(FeatureType::Categorical);
                data = data.with_feature_types(&types).unwrap();
            }
            let ghist = binned(&data, 64);
            let rows = all_rows(n);
            let params = TrainingParams::builder()
                .max_depth(6)
                .alpha(0.1)
                .monotone_constraints(vec![Monotone::None, Monotone::None, Monotone::Increasing])
                .interaction_constraints(vec![vec![0, 1, 2, 3], vec![2, 4, 5, 6, 7]])
                .build()
                .unwrap();
            let builder = HistTreeBuilder::new(&params);
            let new_sampler = || ColumnSampler::new(features, None, 1.0, 0.75, 0.75, 91);
            // Three samplers from one seed, kept in lockstep by the draws below.
            let mut expected_sampler = new_sampler();
            let mut sampler = new_sampler();
            let mut captured_sampler = new_sampler();
            for _ in 0..3 {
                let expected = serial
                    .install(|| builder.build(&ghist, &gradients, &rows, &mut expected_sampler));
                let actual =
                    parallel.install(|| builder.build(&ghist, &gradients, &rows, &mut sampler));
                let (captured, leaves) = parallel.install(|| {
                    builder.build_with_leaf_rows(&ghist, &gradients, &rows, &mut captured_sampler)
                });
                assert_eq!(captured, expected, "{mode}");
                let mut seen = vec![false; n];
                for leaf in leaves {
                    assert!(captured.node(leaf.node).is_leaf());
                    for row in leaf.rows {
                        let row = row as usize;
                        assert!(!seen[row]);
                        seen[row] = true;
                        assert_eq!(
                            leaf.node,
                            captured.leaf_id_with(|f| data.get(row, f as usize))
                        );
                    }
                }
                assert!(seen.into_iter().all(|seen| seen));
                assert!(actual.num_nodes() > 7, "must exercise multiple depths");
                assert_eq!(actual, expected, "{mode}");
                let next = sampler.sample(1);
                assert_eq!(next, expected_sampler.sample(1), "{mode}");
                assert_eq!(next, captured_sampler.sample(1), "{mode}");
            }
        }
    }

    /// Loss-guided growth builds queued nodes' children ahead of their turn
    /// in parallel; the tree and the captured leaf rows must be the serial
    /// ones, including when two queued nodes tie on loss change (the second
    /// half of the rows mirrors the first with negated gradients, so the
    /// root's children score every candidate identically).
    #[test]
    fn parallel_lossguide_grows_the_serial_tree() {
        let half = 6000;
        let features = 6;
        let mut state = 7u64;
        let mut values = Vec::with_capacity(2 * half * features);
        let mut gradients = Vec::with_capacity(2 * half);
        for side in 0..2 {
            for row in 0..half {
                let mut target = 0.0;
                values.push(side as f32);
                for col in 1..features {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let mut value = (state >> 33) as f32 / (1u32 << 31) as f32;
                    target += value * col as f32;
                    if (row * 13 + col * 7) % 11 < 2 {
                        value = f32::NAN;
                    }
                    values.push(value);
                }
                let g = 7.0 - target + 20.0;
                gradients.push(gp(if side == 0 { g } else { -g }, 1.0));
            }
        }
        let n = 2 * half;
        let data = DMatrix::from_dense(&values, n, features).unwrap();
        let ghist = binned(&data, 64);
        let rows = all_rows(n);
        let params = TrainingParams::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .unlimited_depth()
            .max_leaves(24)
            .build()
            .unwrap();
        let builder = HistTreeBuilder::new(&params);
        let grow = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    builder.build_with_leaf_rows(
                        &ghist,
                        &gradients,
                        &rows,
                        &mut ColumnSampler::all(features),
                    )
                })
        };
        let (expected, _) = grow(1);
        assert_eq!(
            expected.node(0).split_feature,
            0,
            "the root separates the halves"
        );
        assert!(expected.num_nodes() > 20);
        for threads in [2, 4, 8] {
            let (tree, leaves) = grow(threads);
            assert_eq!(tree, expected, "{threads} threads");
            let mut seen = vec![false; n];
            for leaf in leaves {
                for row in leaf.rows {
                    assert!(!std::mem::replace(&mut seen[row as usize], true));
                    assert_eq!(
                        leaf.node,
                        tree.leaf_id_with(|f| data.get(row as usize, f as usize))
                    );
                }
            }
            assert!(seen.into_iter().all(|seen| seen));
        }
    }

    #[test]
    fn no_split_below_gamma() {
        let x = vec![0.0f32, 1.0];
        let data = DMatrix::from_dense(&x, 2, 1).unwrap();
        let ghist = binned(&data, 256);
        let gpair = vec![gp(1.0, 1.0), gp(-1.0, 1.0)];
        let params = TrainingParams::builder()
            .max_depth(3)
            .gamma(1e9)
            .build()
            .unwrap();
        let tree = grow_hist(&params, &ghist, &gpair);
        assert_eq!(tree.num_nodes(), 1);
        let (captured, leaves) = HistTreeBuilder::new(&params).build_with_leaf_rows(
            &ghist,
            &gpair,
            &all_rows(2),
            &mut ColumnSampler::all(1),
        );
        assert_eq!(captured, tree);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].node, 0);
        assert_eq!(leaves[0].rows, all_rows(2));
    }

    #[test]
    fn lossguide_respects_max_leaves() {
        // Enough structure that greedy growth would exceed the leaf cap.
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y = Vec::new();
        for i in 0..n {
            y.push(gp(if i % 2 == 0 { 1.0 } else { -1.0 }, 1.0));
        }
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let ghist = binned(&data, 256);
        let params = TrainingParams::builder()
            .grow_policy(GrowPolicy::LossGuide)
            .max_leaves(4)
            .unlimited_depth()
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(0.0)
            .build()
            .unwrap();
        let tree = grow_hist(&params, &ghist, &y);
        assert!(tree.num_leaves() <= 4, "got {} leaves", tree.num_leaves());
    }

    #[test]
    fn monotone_increasing_is_enforced() {
        use crate::config::Monotone;
        let (data, gpair) = monotone_v_shape_data();
        let ghist = binned(&data, 256);
        let params = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(1.0)
            .monotone_constraints(vec![Monotone::Increasing])
            .build()
            .unwrap();
        let tree = grow_hist(&params, &ghist, &gpair);
        // Predictions must be non-decreasing in x under the increasing constraint.
        assert!(non_decreasing(&tree, &data), "monotonicity violated");
    }

    // Collect the split features along every root-to-leaf path.
    fn root_to_leaf_feature_sets(tree: &RegTree) -> Vec<Vec<u32>> {
        fn walk(tree: &RegTree, id: usize, path: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
            let node = tree.node(id);
            if node.is_leaf() {
                out.push(path.clone());
                return;
            }
            path.push(node.split_feature);
            walk(tree, node.left as usize, path, out);
            walk(tree, node.right as usize, path, out);
            path.pop();
        }
        let mut out = Vec::new();
        walk(tree, 0, &mut Vec::new(), &mut out);
        out
    }

    // A 4-feature dataset where every feature carries signal, so an
    // unconstrained tree would happily mix features across groups.
    fn four_feature_data() -> (DMatrix, Vec<GradPair>) {
        let n = 32;
        let mut x = vec![0.0f32; n * 4];
        let mut gpair = Vec::with_capacity(n);
        for i in 0..n {
            // Distinct-ish per-feature patterns so each is individually useful.
            x[i * 4] = (i % 2) as f32;
            x[i * 4 + 1] = (i % 4) as f32;
            x[i * 4 + 2] = (i % 8) as f32;
            x[i * 4 + 3] = (i % 16) as f32;
            let g = if i % 2 == 0 { 1.0 } else { -1.0 };
            gpair.push(gp(g, 1.0));
        }
        (DMatrix::from_dense(&x, n, 4).unwrap(), gpair)
    }

    #[test]
    fn interaction_constraints_confine_paths_to_one_group() {
        let (data, gpair) = four_feature_data();
        let ghist = binned(&data, 256);
        let params = TrainingParams::builder()
            .max_depth(4)
            .min_child_weight(0.0)
            .gamma(0.0)
            .lambda(0.0)
            .interaction_constraints(vec![vec![0, 1], vec![2, 3]])
            .build()
            .unwrap();
        let tree = grow_hist(&params, &ghist, &gpair);

        // Every path's split features must fit inside a single allowed group:
        // never both a {0,1} feature and a {2,3} feature on the same path.
        for path in root_to_leaf_feature_sets(&tree) {
            let has_ab = path.iter().any(|&f| f == 0 || f == 1);
            let has_cd = path.iter().any(|&f| f == 2 || f == 3);
            assert!(
                !(has_ab && has_cd),
                "path mixes interaction groups: {path:?}"
            );
        }
    }

    #[test]
    fn hist_matches_exact_on_small_problem() {
        // Random-ish separable-ish data; hist with enough bins should match exact.
        let n = 60;
        let mut x = Vec::new();
        let mut gpair = Vec::new();
        for i in 0..n {
            let xi = (i as f32) * 0.1;
            x.push(xi);
            gpair.push(gp((xi - 3.0).sin(), 1.0));
        }
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let params = TrainingParams::builder()
            .max_depth(3)
            .lambda(1.0)
            .min_child_weight(1.0)
            .gamma(0.0)
            .build()
            .unwrap();

        let exact = grow_exact(&params, &data, &gpair);

        // 256 bins over 60 distinct values -> each value its own bin -> exact match.
        let ghist = binned(&data, 256);
        let hist = grow_hist(&params, &ghist, &gpair);

        for r in 0..n {
            let pe = exact.predict_row(&data, r);
            let ph = hist.predict_row(&data, r);
            assert!((pe - ph).abs() < 1e-5, "row {r}: exact {pe} vs hist {ph}");
        }
    }

    // Present rows share one gradient sign and missing rows the other. Under an
    // increasing constraint the forward endpoint (present left, missing right)
    // is rejected, and the only split with pure children is XGBoost's backward
    // endpoint: missing mass left, every present bin right.
    #[test]
    fn missing_only_left_split_under_monotone_constraint() {
        use crate::config::Monotone;
        let x = vec![0.0f32, 1.0, 2.0, 3.0, f32::NAN, f32::NAN];
        let n = x.len();
        let data = DMatrix::from_dense(&x, n, 1).unwrap();
        let ghist = binned(&data, 256);
        assert!(ghist.dense_stride().is_none());
        let mut gpair = vec![gp(-1.0, 1.0); 4];
        gpair.extend([gp(1.0, 1.0), gp(1.0, 1.0)]);
        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .monotone_constraints(vec![Monotone::Increasing])
            .build()
            .unwrap();
        let tree = grow_hist(&params, &ghist, &gpair);
        assert_eq!(tree.num_nodes(), 3);
        let root = tree.node(0);
        assert!(root.default_left);
        assert!(root.split_cond.is_finite());
        let left = tree.node(root.left as usize);
        let right = tree.node(root.right as usize);
        assert!(
            (left.sum_hess - 2.0).abs() < 1e-6,
            "left cover {}",
            left.sum_hess
        );
        assert!(
            (right.sum_hess - 4.0).abs() < 1e-6,
            "right cover {}",
            right.sum_hess
        );
        for r in 0..4 {
            assert!(
                (tree.predict_row(&data, r) - 1.0).abs() < 1e-6,
                "present row {r}"
            );
        }
        for r in 4..6 {
            assert!(
                (tree.predict_row(&data, r) + 1.0).abs() < 1e-6,
                "missing row {r}"
            );
        }
    }

    // A sparse index whose split feature is fully present has no missing mass
    // to route: the backward pass never runs, so the split keeps the forward
    // orientation (missing right) even when the tie-breaking alternative
    // would be a missing-left split.
    #[test]
    fn fully_present_feature_never_splits_missing_left() {
        use crate::config::Monotone;
        // Feature 1 carries the NaN that makes the index sparse but is constant
        // otherwise, so only feature 0 can split.
        let x = vec![
            0.0f32,
            5.0,
            1.0,
            5.0,
            2.0,
            5.0,
            3.0,
            f32::NAN,
            4.0,
            5.0,
            5.0,
            5.0,
        ];
        let n = 6;
        let data = DMatrix::from_dense(&x, n, 2).unwrap();
        let ghist = binned(&data, 256);
        assert!(ghist.dense_stride().is_none());
        let gpair = vec![
            gp(1.0, 1.0),
            gp(1.0, 1.0),
            gp(1.0, 1.0),
            gp(-1.0, 1.0),
            gp(-1.0, 1.0),
            gp(-1.0, 1.0),
        ];
        let params = TrainingParams::builder()
            .max_depth(1)
            .lambda(0.0)
            .min_child_weight(0.0)
            .gamma(0.0)
            .monotone_constraints(vec![Monotone::Increasing, Monotone::None])
            .build()
            .unwrap();
        let tree = grow_hist(&params, &ghist, &gpair);
        assert_eq!(tree.num_nodes(), 3);
        let root = tree.node(0);
        assert_eq!(root.split_feature, 0);
        assert!(!root.default_left);
        assert!(
            root.split_cond > 2.0 && root.split_cond <= 3.0,
            "{}",
            root.split_cond
        );
        for r in 0..3 {
            assert!((tree.predict_row(&data, r) + 1.0).abs() < 1e-6, "row {r}");
        }
        for r in 3..6 {
            assert!((tree.predict_row(&data, r) - 1.0).abs() < 1e-6, "row {r}");
        }
    }
}
