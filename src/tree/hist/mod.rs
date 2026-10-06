//! Gradient-histogram construction backends.
//!
//! The [`HistogramBackend`] trait is the single seam a future GPU implementation
//! plugs into: everything above it (the histogram tree builder) is
//! backend-agnostic. The CPU backend uses `rayon` to split the work by
//! feature or into fixed blocks of rows whose partial histograms it reduces
//! in block order, so every histogram is independent of the thread count,
//! and provides the *subtraction trick* (`sibling = parent − child`) that
//! halves histogram construction cost.

pub(crate) mod quantized;
mod walk;

use crate::data::ghist::GHistIndex;
use crate::objective::GradPair;
use crate::tree::gain::{GradStats, RegParams};
use rayon::prelude::*;
use walk::{Bucket, RowValue, SweepRows, accumulate, by_features, contiguous_range};

/// A gradient histogram: one [`GradStats`] bucket per global bin.
pub type Histogram = Vec<GradStats>;

/// Construct a fresh zeroed histogram of the given length.
pub fn zeroed(total_bins: usize) -> Histogram {
    vec![GradStats::default(); total_bins]
}

/// Turn `parent` into the sibling histogram `parent − child` in place. Reusing
/// the parent's buffer avoids allocating and writing a third histogram.
pub fn subtract_in_place(parent: &mut [GradStats], child: &[GradStats]) {
    debug_assert_eq!(parent.len(), child.len());
    for (p, c) in parent.iter_mut().zip(child) {
        *p = p.sub(*c);
    }
}

/// Split a flat histogram of `stride` entries per global bin into each
/// feature's disjoint bin range, in feature order: `(first_bin, slice)`.
pub(crate) fn feature_slices<'a, T>(
    ghist: &GHistIndex,
    hist: &'a mut [T],
    stride: usize,
) -> Vec<(usize, &'a mut [T])> {
    let cuts = ghist.cuts();
    let mut slices = Vec::with_capacity(ghist.n_cols());
    let mut rest = hist;
    let mut next = 0;
    for f in 0..ghist.n_cols() {
        let (fs, fe) = cuts.feature_bins(f);
        assert_eq!(
            fs, next,
            "feature bin ranges must be contiguous and ordered"
        );
        let (head, tail) = rest.split_at_mut((fe - fs) * stride);
        slices.push((fs, head));
        rest = tail;
        next = fe;
    }
    assert!(
        rest.is_empty(),
        "feature bin ranges must cover the histogram"
    );
    slices
}

/// Backend that builds and combines gradient histograms.
///
/// Sibling histograms reuse the parent buffer via the free `subtract_in_place`;
/// there is intentionally no `subtract` hook (a three-slice method would only
/// add an allocation).
pub trait HistogramBackend: Send + Sync {
    /// Accumulate the gradients of `rows` into `out` (length = total bins).
    /// `out` is overwritten (not added to).
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]);

    /// Announce the gradient slice the following [`build`](Self::build) calls
    /// of this tree will read. Called once per tree, before any `build`.
    /// Backends that stage the gradients (a GPU) upload them here; the
    /// default does nothing.
    fn prepare(&self, _ghist: &GHistIndex, _gpair: &[GradPair]) {}

    /// The device-resident row engine of a backend that keeps a tree's
    /// rows on its device (a GPU), letting the hist builder partition and
    /// build whole levels there. `None` (the default): the builder keeps
    /// rows on the host and calls [`build`](Self::build) per node.
    fn row_engine(&self) -> Option<&dyn RowEngine> {
        None
    }
}

/// A node's rows as a [`RowEngine`] keeps them: `len` ascending row ids at
/// `offset` of the engine's row buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

/// Where a split sends a row's present bin of the split feature; a missing
/// value follows the split's `default_left`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    not(all(target_os = "linux", feature = "cuda")),
    allow(dead_code, reason = "only the CUDA row engine reads the rule")
)]
pub enum RowRule<'a> {
    /// Feature-local bins below the limit go left.
    Below(u32),
    /// `left[local bin]`: a categorical split's per-bin direction.
    Table(&'a [bool]),
}

/// One node to partition: its rows and how its split routes them.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    not(all(target_os = "linux", feature = "cuda")),
    allow(dead_code, reason = "only the CUDA row engine reads the split")
)]
pub struct RowSplit<'a> {
    pub(crate) seg: Segment,
    pub(crate) feature: u32,
    pub(crate) rule: RowRule<'a>,
    pub(crate) default_left: bool,
}

/// A partitioned node: the left child holds the segment's first rows, the
/// right child the rest, both in ascending order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Partitioned {
    pub(crate) left: Segment,
    pub(crate) right: Segment,
}

/// Device-resident tree growth ([`HistogramBackend::row_engine`]): a tree's
/// rows live in one device buffer, each node a [`Segment`] of it, and the
/// builder partitions and builds histograms a whole level at a time. Every
/// histogram equals [`CpuBackend::build`]'s of the same rows bit for bit,
/// and partitions are stable, so the tree is the host builder's.
///
/// Every method returns `None` after a device failure; the builder then
/// regrows the tree on the host (or, when the device also holds the
/// gradients, the trainer redoes the round on the host).
pub trait RowEngine: Sync {
    /// Start a tree over `rows` (ascending, distinct), with the gradients
    /// staged ([`HistogramBackend::prepare`], or [`Self::gradients`]):
    /// the root's segment.
    fn begin_tree(&self, ghist: &GHistIndex, rows: &[u32]) -> Option<Segment>;

    /// The statistics of `seg`'s rows, the host `sum_rows`'s blocks summed
    /// on the device (exactly in integers where their sums are exact, else
    /// as `f64` chains) and added in block order: the host's sum bit for
    /// bit. `None` for non-finite gradients.
    fn root_total(&self, seg: Segment) -> Option<GradStats>;

    /// Partition each split's segment in place (stable), in one batch.
    fn partition(&self, ghist: &GHistIndex, splits: &[RowSplit<'_>]) -> Option<Vec<Partitioned>>;

    /// The histograms of the nodes, in one batch. `gpair` is the host copy
    /// of the staged gradients, `None` when only the device has them.
    fn histograms(
        &self,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        nodes: &[Segment],
    ) -> Option<Vec<Histogram>>;

    /// The row ids of each segment.
    fn rows(&self, segs: &[Segment]) -> Option<Vec<Vec<u32>>>;

    /// Keep `margins` (one per row) on the device for device-side rounds.
    fn load_margins(&self, margins: &[f32]) -> Option<()>;

    /// Stage `loss`'s gradients of the device margins for the next tree,
    /// exactly as the host computes them: whether they are all finite (a
    /// tree with non-finite gradients, or rows the device cannot reproduce,
    /// must grow on the host).
    fn gradients(&self, loss: DeviceLoss, labels: &[f32], weights: Option<&[f32]>) -> Option<bool>;

    /// Add each leaf's value to the device margins of its rows.
    fn add_leaf_values(&self, leaves: &[(Segment, f32)]) -> Option<()>;

    /// Copy the device margins into `out`.
    fn read_margins(&self, out: &mut [f32]) -> Option<()>;

    /// Reserve `slots` device histograms for resident growth (histograms
    /// built, subtracted, and searched where they are): `Some(false)`, with
    /// nothing reserved, when they do not fit.
    fn reserve_hists(&self, ghist: &GHistIndex, slots: usize) -> Option<bool>;

    /// Build each `(segment, slot)` node's histogram into its slot, then
    /// turn each `(parent, built)` pair's parent slot into the sibling
    /// `parent - built` ([`subtract_in_place`]). `gpair` as for
    /// [`Self::histograms`].
    fn build_resident(
        &self,
        ghist: &GHistIndex,
        gpair: Option<&[GradPair]>,
        nodes: &[(Segment, HistSlot)],
        siblings: &[(HistSlot, HistSlot)],
    ) -> Option<()>;

    /// The numeric split scan (`scan_numeric_splits` in
    /// `tree::builder::split`) of every request's features on its slot's
    /// histogram, request by request, feature by feature.
    fn scan_resident(
        &self,
        ghist: &GHistIndex,
        reg: &RegParams,
        requests: &[ScanRequest<'_>],
    ) -> Option<Vec<FeatureScan>>;

    /// A slot's histogram, read back.
    fn read_hist(&self, slot: HistSlot) -> Option<Histogram>;
}

/// A [`RowEngine`] histogram slot.
pub type HistSlot = u32;

/// One node's split scan on the device: its histogram's slot, statistics,
/// `root_gain` and monotone bounds (as the scorer uses them, in `f32`), and
/// the numeric features to scan, each with its monotone direction.
#[derive(Debug, Clone, Copy)]
pub struct ScanRequest<'a> {
    pub slot: HistSlot,
    pub total: GradStats,
    pub root_gain: f32,
    pub lower: f32,
    pub upper: f32,
    pub features: &'a [(u32, i8)],
}

/// One feature's numeric split scan, as `scan_numeric_splits` reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureScan {
    /// No candidate has a finite loss change.
    Empty,
    /// Some candidate scored NaN: the host replays the feature.
    Nan,
    /// The first candidate with the largest finite loss change: in the
    /// forward pass (bins `..= offset` left, missing right) or the backward
    /// one (bins `>= offset` right, missing left), with that pass's
    /// accumulated statistics (the left child's forward, the right's
    /// backward).
    Best {
        loss_chg: f32,
        backward: bool,
        offset: u32,
        acc: GradStats,
    },
}

/// A loss whose gradients a [`RowEngine`] computes from its margins with
/// the host's bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceLoss {
    /// `reg:squarederror`.
    SquaredError { scale_pos_weight: f32 },
    /// `reg:logistic`, `binary:logistic`, `binary:logitraw`: the host's
    /// vector kernel over `split.rows`, its scalar path (on the host) after.
    Logistic {
        scale_pos_weight: f32,
        min_hess: f32,
        split: crate::simd::VectorSplit,
    },
}

/// Multi-core CPU histogram backend.
///
/// Each bin's `f64` sum is a function of the node's row count alone, never
/// of the thread count or the index layout, so the serial and parallel
/// builds agree bit for bit and a GPU backend can reproduce them in
/// parallel. A node below 8,192 rows adds each bin's rows in ascending
/// order: the plain chain XGBoost's single-threaded build forms. A larger
/// node sums fixed blocks of about 4,096 rows, each in row order from zero,
/// and adds the block partials to the first block's in block order.
/// Outside the range where `f64` sums are exact (`backend/exact_sum.rs`)
/// that can round differently from the chain, which XGBoost's threaded
/// build does too (its per-thread buffers are reduced in thread order);
/// inside it every grouping gives the chain's sum.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

/// Rows per block of the blocked build, and per task of the quantized one:
/// enough to amortize a partial histogram's zeroing and reduction.
const ROWS_PER_TASK: usize = 4096;
/// Nodes below this many rows are one block (built as a single chain).
pub(crate) const PARALLEL_THRESHOLD: usize = 2 * ROWS_PER_TASK;
/// Bins per task when the partial histograms are summed.
const REDUCE_BINS: usize = 2048;
/// Datasets up to this many rows build row subsets feature by feature: the
/// gradients (8 bytes a row) then fit a core's 2 MiB L2.
const GATHER_MAX_ROWS: usize = 1 << 18;

/// The order in which [`CpuBackend`] adds each bin's rows: a function of
/// the node's row count alone. A device backend reproducing the CPU's `f64`
/// sums bit for bit follows the same order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SumOrder {
    /// One chain in ascending row order, from `+0.0`.
    Chain,
    /// `rows.chunks(grain)`, each chained in row order from `+0.0`; the bin
    /// is the first chunk's partial plus every later one in chunk order.
    /// The chunk count is `rows.len().div_ceil(grain)`, which falls below
    /// `rows.len() / ROWS_PER_TASK` past about 16.8M rows: count chunks from
    /// `grain`, never from the row count.
    Blocked {
        /// Rows per chunk (the last one shorter).
        grain: usize,
    },
}

/// The summation order [`CpuBackend::build`] uses for a node of `len` rows:
/// one chain below [`PARALLEL_THRESHOLD`] rows, else `len / 4096` equal
/// blocks.
pub(crate) fn sum_order(len: usize) -> SumOrder {
    if len < PARALLEL_THRESHOLD {
        return SumOrder::Chain;
    }
    let blocks = len / ROWS_PER_TASK;
    SumOrder::Blocked {
        grain: len.div_ceil(blocks),
    }
}

impl HistogramBackend for CpuBackend {
    fn build(&self, ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair], out: &mut [GradStats]) {
        let SumOrder::Blocked { grain } = sum_order(rows.len()) else {
            out.fill(GradStats::default());
            accumulate(ghist, rows, gpair, out);
            return;
        };
        let threads = rayon::current_num_threads();
        // With a column-major copy, a contiguous row range, or any row
        // subset of an index small enough that every feature group's
        // re-read of the gradients stays in a core's cache, is split by
        // feature; both builds sum the same blocks in the same order.
        let range = contiguous_range(rows);
        let columns = ghist
            .column_bins()
            .filter(|_| threads > 1 && (range.is_some() || ghist.n_rows() <= GATHER_MAX_ROWS));
        if let Some(columns) = columns {
            let rows = match range {
                Some(range) => SweepRows::Range(range),
                None => SweepRows::Subset(rows),
            };
            by_features(ghist, &columns, &rows, gpair, out, grain, threads);
        } else {
            accumulate_blocks(ghist, rows, gpair, out, grain, threads);
        }
    }
}

/// The blocked build of [`CpuBackend`] ([`SumOrder::Blocked`]): `rows`
/// split into `rows.chunks(grain)`, each accumulated from zero into a
/// partial histogram, and `out` = the first partial plus every later one in
/// block order. The blocks depend only on the row count; `threads` only
/// sets how many are built at once (in waves, each reduced into `out`
/// before the next), which bounds the partials held to one per thread.
fn accumulate_blocks(
    ghist: &GHistIndex,
    rows: &[u32],
    gpair: &[GradPair],
    out: &mut [GradStats],
    grain: usize,
    threads: usize,
) {
    let total = out.len();
    let blocks = rows.len().div_ceil(grain);
    let wave = threads.clamp(1, blocks);
    let mut partials: Vec<Histogram> = Vec::with_capacity(wave);
    for (w, wave_rows) in rows.chunks(grain * wave).enumerate() {
        let built = wave_rows.len().div_ceil(grain);
        if w == 0 {
            // Each task allocates (and zeroes) its own partial.
            wave_rows
                .par_chunks(grain)
                .map(|block| {
                    let mut partial = zeroed(total);
                    accumulate(ghist, block, gpair, &mut partial);
                    partial
                })
                .collect_into_vec(&mut partials);
        } else {
            partials
                .par_iter_mut()
                .zip(wave_rows.par_chunks(grain))
                .for_each(|(partial, block)| {
                    partial.fill(GradStats::default());
                    accumulate(ghist, block, gpair, partial);
                });
        }
        // Split by bin range, which keeps each bin's block order while
        // using every worker.
        let (head, rest) = partials[..built].split_at(usize::from(w == 0));
        out.par_chunks_mut(REDUCE_BINS)
            .enumerate()
            .for_each(|(i, out)| {
                let start = i * REDUCE_BINS;
                let end = start + out.len();
                if let [first] = head {
                    out.copy_from_slice(&first[start..end]);
                }
                for partial in rest {
                    for (o, p) in out.iter_mut().zip(&partial[start..end]) {
                        o.add(*p);
                    }
                }
            });
    }
}

/// Bin-index storage widths the accumulation and partition loops specialize on.
pub(crate) trait BinIndex: Copy + Send + Sync {
    fn index(self) -> usize;
}

impl BinIndex for u16 {
    #[inline(always)]
    fn index(self) -> usize {
        self as usize
    }
}

impl BinIndex for u32 {
    #[inline(always)]
    fn index(self) -> usize {
        self as usize
    }
}

impl Bucket for GradStats {
    #[inline(always)]
    fn push(&mut self, value: GradStats) {
        self.add(value);
    }
}

impl RowValue<GradStats> for GradPair {
    #[inline(always)]
    fn value(self) -> GradStats {
        GradStats::from_pair(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DMatrix;
    use crate::data::ghist::Bins;
    use crate::data::quantile::HistCuts;

    #[test]
    fn subtraction_identity() {
        // parent = left + right, so parent - left = right.
        let total = 8;
        let mut parent = zeroed(total);
        let mut left = zeroed(total);
        let mut right = zeroed(total);
        for i in 0..total {
            left[i] = GradStats::new(i as f64, 1.0);
            right[i] = GradStats::new(-(i as f64) * 0.5, 2.0);
            parent[i] = GradStats::new(left[i].grad + right[i].grad, left[i].hess + right[i].hess);
        }
        let mut out = parent.clone();
        subtract_in_place(&mut out, &left);
        for i in 0..total {
            assert!((out[i].grad - right[i].grad).abs() < 1e-12);
            assert!((out[i].hess - right[i].hess).abs() < 1e-12);
        }
    }

    /// Row-major reference independent of `accumulate`: every bin receives
    /// its rows in ascending order.
    fn row_order_reference(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair]) -> Histogram {
        let stride = ghist.dense_stride().expect("dense index");
        let mut h = zeroed(ghist.total_bins());
        for &r in rows {
            let r = r as usize;
            let g = GradStats::from_pair(gpair[r]);
            for f in 0..stride {
                let bin = match ghist.bins() {
                    Bins::U16(b) => b[r * stride + f] as usize,
                    Bins::U32(b) => b[r * stride + f] as usize,
                };
                h[bin].add(g);
            }
        }
        h
    }

    /// The block-order reference of a dense index: [`row_order_reference`]
    /// of each `grain`-row chunk, the partials added in chunk order to the
    /// first one.
    fn block_order_reference(ghist: &GHistIndex, rows: &[u32], gpair: &[GradPair]) -> Histogram {
        let SumOrder::Blocked { grain } = sum_order(rows.len()) else {
            return row_order_reference(ghist, rows, gpair);
        };
        let mut chunks = rows.chunks(grain);
        let mut h = row_order_reference(ghist, chunks.next().unwrap_or(&[]), gpair);
        for chunk in chunks {
            for (o, p) in h.iter_mut().zip(row_order_reference(ghist, chunk, gpair)) {
                o.add(p);
            }
        }
        h
    }

    /// On a dense index the feature-parallel sweep (several threads) and
    /// the row-blocked build (one thread) both give the block-order
    /// reference bit for bit, for contiguous ranges and row subsets, on
    /// gradients spanning enough exponents that the grouping shows.
    #[test]
    fn dense_builds_match_the_block_order_reference() {
        let (n, f) = (3 * PARALLEL_THRESHOLD + 129, 7);
        let x: Vec<f32> = (0..n * f)
            .map(|i| ((i * 2_654_435_761_usize) % 1009) as f32 / 7.0)
            .collect();
        let data = DMatrix::from_dense(&x, n, f).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 64);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        assert!(
            ghist.column_bins().is_some(),
            "dense index keeps a column copy"
        );
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| {
                let scale = 2f32.powi((i * 37 % 61) as i32 - 30);
                GradPair::new(
                    ((i * 7919) % 1237) as f32 / 331.0 * scale - 1.9,
                    0.25 + (i % 5) as f32,
                )
            })
            .collect();
        let bits = |h: &Histogram| -> Vec<(u64, u64)> {
            h.iter()
                .map(|s| (s.grad.to_bits(), s.hess.to_bits()))
                .collect()
        };
        let all: Vec<u32> = (0..n as u32).collect();
        let subset: Vec<u32> = (0..n as u32).filter(|r| r % 3 != 1).collect();
        let offset: Vec<u32> = (1000..(1000 + PARALLEL_THRESHOLD) as u32).collect();
        assert!(contiguous_range(&all).is_some() && contiguous_range(&offset).is_some());
        assert!(contiguous_range(&subset).is_none());
        assert!(contiguous_range(&[2, 0, 1]).is_none() && contiguous_range(&[5, 5]).is_none());
        assert_ne!(
            bits(&block_order_reference(&ghist, &all, &gpair)),
            bits(&row_order_reference(&ghist, &all, &gpair)),
            "the case must separate the groupings"
        );
        for rows in [&all, &subset, &offset] {
            let expect = bits(&block_order_reference(&ghist, rows, &gpair));
            for threads in [1, 4] {
                let mut out = zeroed(ghist.total_bins());
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| CpuBackend.build(&ghist, rows, &gpair, &mut out));
                assert_eq!(bits(&out), expect, "rows={} threads={threads}", rows.len());
            }
        }
    }

    /// A sparse index of `n` rows holding only feature 0 (of 3), binned from
    /// `x`.
    fn sparse_index(x: &[f32]) -> GHistIndex {
        let n = x.len();
        let data = DMatrix::from_csr((0..=n).collect(), vec![0; n], x.to_vec(), 3).unwrap();
        let cuts = HistCuts::from_dmatrix(&data, 256);
        let ghist = GHistIndex::from_dmatrix(&data, cuts);
        assert!(
            ghist.column_bins().is_none(),
            "sparse index has no column copy"
        );
        ghist
    }

    fn build_on(
        threads: usize,
        ghist: &GHistIndex,
        rows: &[u32],
        gpair: &[GradPair],
    ) -> Vec<(f64, f64)> {
        let mut out = zeroed(ghist.total_bins());
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| CpuBackend.build(ghist, rows, gpair, &mut out));
        out.iter().map(|s| (s.grad, s.hess)).collect()
    }

    /// The reported case: 8,192 sparse rows in one bin, gradients `2^54` at
    /// row 0, `-2^54` at row 4096, `1` at row 4097. The row-order chain sums
    /// to `1`; the two blocks of 4,096 rows sum to `2^54` and
    /// `-2^54 + 1 = -2^54` (rounded), so `0`. Every thread count, one
    /// included, builds the blocks.
    #[test]
    fn histograms_do_not_depend_on_the_thread_count() {
        let n = 2 * ROWS_PER_TASK;
        let ghist = sparse_index(&vec![1.0; n]);
        assert_eq!(
            ghist.cuts().feature_bins(0),
            (0, 1),
            "feature 0 has one bin"
        );
        let mut gpair = vec![GradPair::new(0.0, 1.0); n];
        gpair[0].grad = 2f32.powi(54);
        gpair[4096].grad = -(2f32.powi(54));
        gpair[4097].grad = 1.0;
        let rows: Vec<u32> = (0..n as u32).collect();
        for threads in [1, 2, 4, 8] {
            let hist = build_on(threads, &ghist, &rows, &gpair);
            assert_eq!(hist[0], (0.0, n as f64), "{threads} threads");
            assert!(
                hist[1..].iter().all(|&b| b == (0.0, 0.0)),
                "{threads} threads"
            );
        }
    }

    /// The blocked build over several waves (five blocks, built two or
    /// three at a time) equals an independent block-order reference for
    /// every thread count, on gradients spanning enough exponents that the
    /// grouping shows in the low bits.
    #[test]
    fn blocked_build_matches_the_block_order_reference() {
        let n = 6 * ROWS_PER_TASK + 123;
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let ghist = sparse_index(&x);
        let gpair: Vec<GradPair> = (0..n)
            .map(|i| {
                let scale = 2f32.powi((i * 37 % 61) as i32 - 30);
                GradPair::new(((i * 7919) % 1237) as f32 / 331.0 * scale - 1.0, scale)
            })
            .collect();
        let rows: Vec<u32> = (0..n as u32).filter(|r| r % 11 != 3).collect();
        let grain = rows.len().div_ceil(rows.len() / ROWS_PER_TASK);
        let mut expect = zeroed(ghist.total_bins());
        for block in rows.chunks(grain) {
            let mut partial = zeroed(ghist.total_bins());
            for &r in block {
                let bin = match ghist.bins() {
                    Bins::U16(b) => usize::from(b[ghist.row_ptr()[r as usize]]),
                    Bins::U32(b) => b[ghist.row_ptr()[r as usize]] as usize,
                };
                partial[bin].add(GradStats::from_pair(gpair[r as usize]));
            }
            for (e, p) in expect.iter_mut().zip(&partial) {
                e.add(*p);
            }
        }
        let expect: Vec<(f64, f64)> = expect.iter().map(|s| (s.grad, s.hess)).collect();
        let chain: Vec<(f64, f64)> = {
            let mut h = zeroed(ghist.total_bins());
            accumulate(&ghist, &rows, &gpair, &mut h);
            h.iter().map(|s| (s.grad, s.hess)).collect()
        };
        assert_ne!(expect, chain, "the case must separate the groupings");
        for threads in [1, 2, 3, 8] {
            assert_eq!(
                build_on(threads, &ghist, &rows, &gpair),
                expect,
                "{threads} threads"
            );
        }
    }
}
