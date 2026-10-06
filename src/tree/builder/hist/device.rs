//! Device-resident growth: a [`RowEngine`] keeps the tree's rows on its
//! device, so each depthwise level (or loss-guided expansion) partitions
//! every splitting node and builds every needed child histogram in one
//! batch there. The host keeps everything that decides the tree — node
//! order, the column sampler, bounds, interaction state, and the merge of
//! the split search — so the tree is the host builder's, bit for bit.
//!
//! Under *resident* growth (depthwise, numeric features only) the
//! histograms never leave the device: they live in the engine's slots, the
//! device subtracts each sibling from its parent and runs every feature's
//! numeric scan (`scan_numeric_splits`), and the host merges the
//! per-feature results in feature order with XGBoost's tie rule, as its own
//! search does. A node whose scan saw a NaN loss change is searched on the
//! host from its histogram, read back. Otherwise the histograms are
//! downloaded and searched on the host.

use super::search::{permitted, plain_numeric};
use super::{Child, HistTreeBuilder, NodeCtx, NodeEntry, NodeStore, PendingSplit};
use crate::config::GrowPolicy;
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::objective::GradPair;
use crate::tree::builder::partition::{category_left, with_sibling};
use crate::tree::builder::shared::{InteractionState, LeafRows, rayon_available, sum_rows};
use crate::tree::builder::split::NumericScan;
use crate::tree::builder::{BestSplit, Children, SplitLocation, SplitPos, limit_or_unbounded};
use crate::tree::gain::GradStats;
use crate::tree::hist::{
    FeatureScan, HistSlot, Partitioned, RowEngine, RowRule, RowSplit, ScanRequest, Segment,
};
use crate::tree::regtree::RegTree;
use crate::tree::sampler::ColumnSampler;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::BinaryHeap;

/// Combined rows of a batch of splits at which their children's split
/// searches run in parallel.
const PARALLEL_FINISH_ROWS: usize = 4096;

/// What every device-growth step reads: the engine holding the rows, the
/// binned index, and the host copy of the tree's gradients (`None` when
/// only the device has them).
#[derive(Clone, Copy)]
struct Device<'a> {
    engine: &'a dyn RowEngine,
    ghist: &'a GHistIndex,
    gpair: Option<&'a [GradPair]>,
}

/// What a device-grown tree reports about its leaves' rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LeafReport {
    /// Nothing (the rows are not needed).
    Nothing,
    /// Each leaf's rows, read back to the host.
    Rows,
    /// Each leaf's device segment (the rows stay on the device).
    Segments,
}

/// A device-grown tree, its leaves' rows (when read back), and its leaves'
/// device segments (when kept on the device), each with its node id.
type DeviceTree = (RegTree, Vec<LeafRows>, Vec<(usize, Segment)>);

/// The histogram slots of a resident tree: each non-terminal split takes
/// the next one for its built child (the sibling keeps the parent's).
struct Slots {
    next: HistSlot,
    reserved: usize,
}

impl Slots {
    fn take(&mut self) -> Option<HistSlot> {
        let slot = self.next;
        ((slot as usize) < self.reserved).then(|| {
            self.next += 1;
            slot
        })
    }
}

/// One node's resident split search: its histogram's slot, its sampled
/// features, interaction state, and context.
struct ResidentNode<'n> {
    slot: HistSlot,
    features: &'n [u32],
    allowed: Option<&'n InteractionState>,
    ctx: NodeCtx,
}

impl HistTreeBuilder<'_> {
    /// The backend's row engine, when this tree can grow on it: depthwise
    /// or loss-guided growth on `f64` histograms, without reuse penalties
    /// (whose commits a device failure's regrowth would repeat).
    pub(super) fn device_engine(&self) -> Option<&dyn RowEngine> {
        let params = self.config.params;
        let supported = params.quantized.is_none()
            && self.reuse.is_none()
            && matches!(
                params.grow_policy,
                GrowPolicy::DepthWise | GrowPolicy::LossGuide
            );
        supported.then(|| self.backend.row_engine()).flatten()
    }

    /// The slots a resident tree over `rows` rows needs, when it can grow
    /// resident: depthwise to a depth limit, without LightGBM split options
    /// or categorical features. Each non-terminal split takes one slot, so
    /// a tree of depth `d` needs at most `2^(d-1)` (the root's and one per
    /// split above the last level), and never more than its rows.
    fn resident_slots(&self, ghist: &GHistIndex, rows: usize) -> Option<usize> {
        let params = self.config.params;
        let cuts = ghist.cuts();
        if params.grow_policy != GrowPolicy::DepthWise
            || self.options.is_some()
            || (0..ghist.n_cols()).any(|f| cuts.is_categorical(f))
        {
            return None;
        }
        let depth = params.max_depth?.get();
        let by_depth = 1usize.checked_shl(u32::try_from(depth - 1).ok()?)?;
        Some(by_depth.min(rows.max(1)))
    }

    /// Grow the tree on `engine`; `None` after a device failure (the
    /// caller regrows it on the host from the sampler's starting state, or
    /// redoes the round when `gpair` is `None`: the gradients then exist
    /// only on the device).
    pub(super) fn build_on_device(
        &self,
        engine: &dyn RowEngine,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        row_subset: &[u32],
        sampler: &mut ColumnSampler,
        report: LeafReport,
    ) -> Option<DeviceTree> {
        let seg = engine.begin_tree(ghist, row_subset)?;
        // The root's statistics, the host's `sum_rows` bit for bit (on the
        // host only for non-finite gradients).
        let root_stats = match (engine.root_total(seg), gpair) {
            (Some(stats), _) => stats,
            (None, Some(gpair)) => sum_rows(gpair, row_subset),
            (None, None) => return None,
        };
        let on = Device {
            engine,
            ghist,
            gpair,
        };
        let mut slots = self
            .resident_slots(ghist, row_subset.len())
            .filter(|&reserved| engine.reserve_hists(ghist, reserved) == Some(true))
            .map(|reserved| Slots { next: 0, reserved });
        let root = if let Some(slots) = &mut slots {
            let slot = slots.take()?;
            engine.build_resident(ghist, gpair, &[(seg, slot)], &[])?;
            let (features, ctx) = self.root_context(sampler, root_stats, row_subset.len());
            let node = ResidentNode {
                slot,
                features: &features,
                allowed: None,
                ctx,
            };
            let best = self.resident_search(on, &[node])?.pop()?;
            NodeEntry {
                seg: Some(seg),
                slot: Some(slot),
                ..Self::root_node(ctx.tree_seed, best)
            }
        } else {
            let root_hist = engine
                .histograms(ghist, gpair, &[seg])?
                .into_iter()
                .next()?;
            NodeEntry {
                seg: Some(seg),
                ..self.root_entry(ghist, sampler, root_stats, root_hist, row_subset.len())
            }
        };
        let mut tree = RegTree::with_root(root_stats.hess as f32);
        let mut store = NodeStore::new(root_stats, report != LeafReport::Nothing);
        if self.config.params.grow_policy == GrowPolicy::LossGuide {
            self.grow_device_lossguide(on, &mut tree, &mut store, sampler, root)?;
        } else {
            self.grow_device_depthwise(on, &mut tree, &mut store, sampler, root, slots.as_mut())?;
        }
        self.finish_tree(&mut tree, &store, root_stats);
        let leaf_rows = if report == LeafReport::Rows {
            let segs: Vec<_> = store.device_leaves.iter().map(|&(_, seg)| seg).collect();
            let rows = engine.rows(&segs)?;
            store
                .device_leaves
                .iter()
                .zip(rows)
                .map(|(&(node, _), rows)| LeafRows { node, rows })
                .collect()
        } else {
            Vec::new()
        };
        let segments = if report == LeafReport::Segments {
            store.device_leaves
        } else {
            Vec::new()
        };
        Some((tree, leaf_rows, segments))
    }

    /// Grow a tree on the device whose gradients only the device holds
    /// ([`RowEngine::gradients`]): the tree (leaves not yet scaled by
    /// the learning rate) and each leaf's device segment. `None` when the
    /// backend has no row engine for this configuration, or after a device
    /// failure; the trainer then redoes the round on the host.
    pub(crate) fn build_device_staged(
        &self,
        ghist: &GHistIndex,
        rows: &[u32],
        sampler: &mut ColumnSampler,
    ) -> Option<(RegTree, Vec<(usize, Segment)>)> {
        let engine = self.device_engine()?;
        self.build_on_device(engine, ghist, None, rows, sampler, LeafReport::Segments)
            .map(|(tree, _, segments)| (tree, segments))
    }

    fn grow_device_depthwise(
        &self,
        on: Device<'_>,
        tree: &mut RegTree,
        store: &mut NodeStore,
        sampler: &mut ColumnSampler,
        root: NodeEntry,
        mut slots: Option<&mut Slots>,
    ) -> Option<()> {
        let ghist = on.ghist;
        let limit = limit_or_unbounded(self.config.params.max_depth);
        let mut frontier = vec![root];
        let mut depth = 0;
        while depth < limit && !frontier.is_empty() {
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
            let children = match slots.as_deref_mut() {
                Some(slots) => self.resident_children(on, pending, slots)?,
                None => self.device_children(on, pending)?,
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
        Some(())
    }

    fn grow_device_lossguide(
        &self,
        on: Device<'_>,
        tree: &mut RegTree,
        store: &mut NodeStore,
        sampler: &mut ColumnSampler,
        root: NodeEntry,
    ) -> Option<()> {
        let ghist = on.ghist;
        let limit = limit_or_unbounded(self.config.params.max_depth);
        let max_leaves = limit_or_unbounded(self.config.params.max_leaves);
        let mut heap = BinaryHeap::new();
        heap.push(root);
        let mut n_leaves = 1usize;
        while let Some(entry) = heap.pop() {
            if n_leaves >= max_leaves {
                store.record_leaf(entry);
                break;
            }
            if entry.depth >= limit || !self.valid(&entry.best) {
                store.record_leaf(entry);
                continue;
            }
            let split = self.prepare_split(tree, store, ghist.cuts(), sampler, entry);
            n_leaves += 1;
            if let Some(split) = split {
                let (left, right) = self.device_children(on, vec![split])?.pop()?;
                heap.push(left);
                heap.push(right);
            }
        }
        for entry in heap {
            store.record_leaf(entry);
        }
        Some(())
    }

    /// One device partition of every split of `pending`, in order.
    fn partition_pending(on: Device<'_>, pending: &[PendingSplit]) -> Option<Vec<Partitioned>> {
        let cuts = on.ghist.cuts();
        let tables: Vec<Option<Vec<bool>>> = pending
            .iter()
            .map(|split| match &split.entry.best.location {
                SplitLocation::Categories(categories) => {
                    Some(category_left(cuts, split.entry.best.feature, categories))
                }
                SplitLocation::Numeric(_) => None,
            })
            .collect();
        let splits: Option<Vec<RowSplit<'_>>> = pending
            .iter()
            .zip(&tables)
            .map(|(split, table)| {
                let best = &split.entry.best;
                let (fs, _) = cuts.feature_bins(best.feature as usize);
                let rule = match (&best.location, table) {
                    (_, Some(table)) => RowRule::Table(table),
                    (SplitLocation::Numeric(pos), None) => {
                        // Present bins up to the split bin go left.
                        let limit = pos.bin().map_or(0, |s| (s + 1).saturating_sub(fs));
                        RowRule::Below(u32::try_from(limit).ok()?)
                    }
                    (SplitLocation::Categories(_), None) => return None,
                };
                Some(RowSplit {
                    seg: split.entry.seg?,
                    feature: best.feature,
                    rule,
                    default_left: best.default_left,
                })
            })
            .collect();
        on.engine.partition(on.ghist, &splits?)
    }

    /// The children of every split of `pending`: one device partition of
    /// all of them, one batch of the smaller children's histograms (the
    /// siblings by subtraction here), then their split searches.
    fn device_children(
        &self,
        on: Device<'_>,
        mut pending: Vec<PendingSplit>,
    ) -> Option<Vec<(NodeEntry, NodeEntry)>> {
        let Device {
            engine,
            ghist,
            gpair,
        } = on;
        if pending.is_empty() {
            return Some(Vec::new());
        }
        let parts = Self::partition_pending(on, &pending)?;

        // The smaller child of every non-terminal split is built; its
        // sibling is the parent minus it, as the host builder does.
        let built: Vec<Segment> = pending
            .iter()
            .zip(&parts)
            .filter(|(split, _)| !split.terminal)
            .map(|(_, part)| smaller(part))
            .collect();
        let mut hists = engine.histograms(ghist, gpair, &built)?.into_iter();
        let mut items = Vec::with_capacity(pending.len());
        for (split, part) in pending.iter_mut().zip(&parts) {
            let parent = std::mem::take(&mut split.entry.hist);
            let (left_hist, right_hist) = if split.terminal {
                (Vec::new(), Vec::new())
            } else {
                let left_smaller = part.left.len <= part.right.len;
                with_sibling(parent, hists.next()?, left_smaller)
            };
            items.push((part, left_hist, right_hist));
        }
        let finish = |(split, (part, left_hist, right_hist)): (PendingSplit, (&_, _, _))| {
            let part: &Partitioned = part;
            self.finish_children(
                ghist,
                split,
                device_child(part.left, left_hist),
                device_child(part.right, right_hist),
                None,
            )
        };
        let rows: usize = parts.iter().map(|p| p.left.len + p.right.len).sum();
        let pairs = pending.into_iter().zip(items);
        Some(
            if pairs.len() > 1 && rows >= PARALLEL_FINISH_ROWS && rayon_available() {
                pairs
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(finish)
                    .collect()
            } else {
                pairs.map(finish).collect()
            },
        )
    }

    /// [`Self::device_children`] under resident growth: the smaller child
    /// of each non-terminal split is built into a new slot, its sibling
    /// subtracted in the parent's, and every child searched on the device.
    fn resident_children(
        &self,
        on: Device<'_>,
        pending: Vec<PendingSplit>,
        slots: &mut Slots,
    ) -> Option<Vec<(NodeEntry, NodeEntry)>> {
        if pending.is_empty() {
            return Some(Vec::new());
        }
        let parts = Self::partition_pending(on, &pending)?;
        let mut build = Vec::new();
        let mut siblings = Vec::new();
        // Each split's `(left, right)` child slots; `None` when terminal.
        let mut child_slots = Vec::with_capacity(pending.len());
        for (split, part) in pending.iter().zip(&parts) {
            let parent = split.entry.slot?;
            if split.terminal {
                child_slots.push(None);
                continue;
            }
            let built = slots.take()?;
            build.push((smaller(part), built));
            siblings.push((parent, built));
            child_slots.push(Some(if part.left.len <= part.right.len {
                (built, parent)
            } else {
                (parent, built)
            }));
        }
        on.engine
            .build_resident(on.ghist, on.gpair, &build, &siblings)?;
        let contexts: Vec<_> = pending
            .iter()
            .zip(&parts)
            .map(|(split, part)| self.child_contexts(split, part.left.len, part.right.len))
            .collect();
        let mut nodes = Vec::new();
        for ((split, (allowed, left_ctx, right_ctx)), slots) in
            pending.iter().zip(&contexts).zip(&child_slots)
        {
            if let Some((left, right)) = *slots {
                nodes.push(ResidentNode {
                    slot: left,
                    features: &split.left_features,
                    allowed: allowed.as_ref(),
                    ctx: *left_ctx,
                });
                nodes.push(ResidentNode {
                    slot: right,
                    features: &split.right_features,
                    allowed: allowed.as_ref(),
                    ctx: *right_ctx,
                });
            }
        }
        let mut bests = self.resident_search(on, &nodes)?.into_iter();
        let mut out = Vec::with_capacity(pending.len());
        for ((split, part), slots) in pending.into_iter().zip(&parts).zip(child_slots) {
            let searched = match slots {
                Some(_) => Some((bests.next()?, bests.next()?)),
                None => None,
            };
            let (left, right) = self.finish_children(
                on.ghist,
                split,
                device_child(part.left, Vec::new()),
                device_child(part.right, Vec::new()),
                searched,
            );
            out.push(match slots {
                Some((left_slot, right_slot)) => (
                    NodeEntry {
                        slot: Some(left_slot),
                        ..left
                    },
                    NodeEntry {
                        slot: Some(right_slot),
                        ..right
                    },
                ),
                None => (left, right),
            });
        }
        Some(out)
    }

    /// The best split of each resident node: every permitted numeric
    /// feature scanned on the device, merged here in feature order; a node
    /// whose merge needs its histogram (a NaN scan) is searched on the host
    /// from the histogram read back.
    fn resident_search(
        &self,
        on: Device<'_>,
        nodes: &[ResidentNode<'_>],
    ) -> Option<Vec<BestSplit>> {
        let cuts = on.ghist.cuts();
        let features: Vec<Cow<'_, [u32]>> = nodes
            .iter()
            .map(|node| permitted(node.features, node.allowed))
            .collect();
        let scanned: Vec<Vec<(u32, i8)>> = features
            .iter()
            .map(|features| {
                features
                    .iter()
                    .copied()
                    .filter(|&f| plain_numeric(cuts, f))
                    .map(|f| (f, self.config.cons.dir(f as usize)))
                    .collect()
            })
            .collect();
        let requests: Vec<ScanRequest<'_>> = nodes
            .iter()
            .zip(&scanned)
            .map(|(node, features)| ScanRequest {
                slot: node.slot,
                total: node.ctx.stats,
                root_gain: self.node_scorer(node.ctx).root_gain,
                lower: node.ctx.bounds.lower as f32,
                upper: node.ctx.bounds.upper as f32,
                features,
            })
            .collect();
        let results = on
            .engine
            .scan_resident(on.ghist, &self.config.reg, &requests)?;
        if results.len() != scanned.iter().map(Vec::len).sum::<usize>() {
            return None;
        }
        let mut results = results.into_iter();
        let scans: Vec<Vec<Option<NumericScan>>> = features
            .iter()
            .zip(nodes)
            .map(|(features, node)| {
                features
                    .iter()
                    .map(|&f| {
                        if !plain_numeric(cuts, f) {
                            return None;
                        }
                        let scan = results.next()?;
                        Some(numeric_scan(scan, cuts, f, node.ctx.stats))
                    })
                    .collect()
            })
            .collect();
        let items = nodes.iter().zip(&features).zip(scans);
        if nodes.len() > 1 && rayon_available() {
            items
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|((node, features), scans)| self.resident_merge(on, node, features, scans))
                .collect()
        } else {
            items
                .map(|((node, features), scans)| self.resident_merge(on, node, features, scans))
                .collect()
        }
    }

    /// `node`'s best split from its device scans (by position in its
    /// permitted `features`), or from the host search on its histogram when
    /// the merge needs it.
    fn resident_merge(
        &self,
        on: Device<'_>,
        node: &ResidentNode<'_>,
        features: &[u32],
        scans: Vec<Option<NumericScan>>,
    ) -> Option<BestSplit> {
        if let Some(best) = self.merge(on.ghist, None, features, node.ctx, Some(scans)) {
            return Some(best);
        }
        let hist = on.engine.read_hist(node.slot)?;
        Some(self.evaluate(on.ghist, &hist, node.features, node.allowed, node.ctx))
    }
}

/// The child of a partition with fewer rows (the left one on a tie), whose
/// histogram is built.
fn smaller(part: &Partitioned) -> Segment {
    if part.left.len <= part.right.len {
        part.left
    } else {
        part.right
    }
}

/// A child whose rows are the device segment `seg`.
fn device_child(seg: Segment, hist: Vec<GradStats>) -> Child {
    Child {
        len: seg.len,
        rows: Vec::new(),
        seg: Some(seg),
        hist,
        quant: None,
    }
}

/// A device scan of feature `f` (whose node has statistics `total`) as the
/// host's [`NumericScan`].
fn numeric_scan(scan: FeatureScan, cuts: &HistCuts, f: u32, total: GradStats) -> NumericScan {
    match scan {
        FeatureScan::Empty => NumericScan::Empty,
        FeatureScan::Nan => NumericScan::Nan,
        FeatureScan::Best {
            loss_chg,
            backward,
            offset,
            acc,
        } => {
            let (first, _) = cuts.feature_bins(f as usize);
            let offset = offset as usize;
            let (pos, children) = if backward {
                (
                    SplitPos::backward(first, offset),
                    Children::new(true, total.sub(acc), acc),
                )
            } else {
                (
                    SplitPos::Bin(first + offset),
                    Children::new(false, acc, total.sub(acc)),
                )
            };
            NumericScan::Best {
                loss_chg,
                pos,
                children,
            }
        }
    }
}
