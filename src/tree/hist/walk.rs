//! The traversal the `f64` and quantized histograms share: which layout a
//! build sweeps (column-major copy, dense row tiles, or CSR rows), in which
//! order its rows reach each bin, and what it prefetches. Both histograms
//! instantiate the same sequential loops ([`accumulate`]), monomorphized
//! over a [`Bucket`] (the histogram entry: `GradStats`, or a packed integer)
//! and a [`RowValue`] (a row's stored statistic, turned into the value it
//! adds to each of its bins).
//!
//! Every traversal adds each bin's rows in the order of `rows`, ascending,
//! so the `f64` sums are the row-order chain; tiling and grouping change
//! only which bins are hot.

use super::{BinIndex, feature_slices};
use crate::data::ghist::{Bins, GHistIndex};
use rayon::prelude::*;
use std::ops::Range;

/// A histogram bin's running sum.
pub(super) trait Bucket: Copy + Default + Send + Sync {
    /// Add one row's value.
    fn push(&mut self, value: Self);
}

/// A row's stored statistic (the slices the traversal reads are indexed by
/// row), turned into the value it adds to each of its bins.
pub(super) trait RowValue<E: Bucket>: Copy + Sync {
    fn value(self) -> E;
}

/// Rows to run ahead of the accumulation loop when prefetching. Each row's bins
/// and value are fetched into L1 before the loop needs them. Subsets deep in
/// the tree are too sparse for hardware stride prediction.
const PREFETCH_ROWS: usize = 8;
const CACHE_LINE: usize = 64;
/// Rows per tile of the dense accumulation. A tile's row lines stay in cache while
/// its feature blocks are swept, so re-reading them per block is cheap.
const TILE_ROWS: usize = 1024;
/// Histogram bytes a feature block may span, so the block's histogram slice
/// stays resident in a 64 KiB L1 while a tile is accumulated into it (4,096
/// `f64` bins).
const BLOCK_BYTES: usize = 64 * 1024;

/// Sequential accumulation of `rows` into `out` (added, not reset), `values`
/// indexed by row. Specialized on the bin-index width so the inner loop reads
/// the narrowest integers.
#[inline]
pub(super) fn accumulate<E: Bucket, V: RowValue<E>>(
    ghist: &GHistIndex,
    rows: &[u32],
    values: &[V],
    out: &mut [E],
) {
    match ghist.bins() {
        Bins::U16(bins) => accumulate_bins(ghist, bins, rows, values, out),
        Bins::U32(bins) => accumulate_bins(ghist, bins, rows, values, out),
    }
}

/// Prefetch the `len` bins of one row starting at `start`, a cache line at a
/// time.
#[inline(always)]
fn prefetch_bins<B>(bins: &[B], start: usize, len: usize) {
    for offset in (0..len).step_by(CACHE_LINE / std::mem::size_of::<B>()) {
        if let Some(bin) = bins.get(start + offset) {
            crate::simd::prefetch_read(bin);
        }
    }
}

/// Feature blocks `[f0, f1)` of a dense index with `stride` features whose
/// bin ranges each span at most `block_bins` (a single feature may exceed it).
fn feature_blocks(ghist: &GHistIndex, stride: usize, block_bins: usize) -> Vec<(usize, usize)> {
    let cuts = ghist.cuts();
    let mut blocks = Vec::new();
    let mut block_start = 0;
    for f in 1..=stride {
        let span = cuts.feature_bins(f - 1).1 - cuts.feature_bins(block_start).0;
        if span > block_bins && f - 1 > block_start {
            blocks.push((block_start, f - 1));
            block_start = f - 1;
        }
    }
    blocks.push((block_start, stride));
    blocks
}

#[inline(always)]
fn accumulate_bins<E: Bucket, V: RowValue<E>, B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    rows: &[u32],
    values: &[V],
    out: &mut [E],
) {
    // Establishes the bound used by `add_row`: with `out` covering every bin,
    // the `GHistIndex` invariant (all stored bins < total_bins) makes every
    // histogram index in range.
    assert_eq!(
        out.len(),
        ghist.total_bins(),
        "histogram length must equal the binned index's bin count"
    );
    // A row holds at most one bin per feature (dense rows one each, and
    // `DMatrix` refuses duplicate CSR columns), from disjoint ranges, so its
    // bins are distinct: four histogram entries are loaded before any is
    // stored, which lets the loads overlap. Each bin still receives its rows
    // in order.
    let add_row = |row_bins: &[B], v: V, out: &mut [E]| {
        let v = v.value();
        let base = out.as_mut_ptr();
        let (quads, rest) = row_bins.as_chunks::<4>();
        for &quad in quads {
            let [a, b, c, d] = quad.map(BinIndex::index);
            // SAFETY: every bin is `< ghist.total_bins() == out.len()` by the
            // index invariant and the assertion above, and the four bins are
            // distinct (different features of one row), so the reads and
            // writes are in bounds and never overlap.
            unsafe {
                let (ha, hb, hc, hd) = (*base.add(a), *base.add(b), *base.add(c), *base.add(d));
                let add = |mut h: E| {
                    h.push(v);
                    h
                };
                *base.add(a) = add(ha);
                *base.add(b) = add(hb);
                *base.add(c) = add(hc);
                *base.add(d) = add(hd);
            }
        }
        for &bin in rest {
            // SAFETY: as above.
            unsafe { &mut *base.add(bin.index()) }.push(v);
        }
    };

    // A contiguous row range (the root, or a root chunk, without row
    // sampling) sweeps the column-major copy one feature at a time: the bins
    // stream sequentially and each feature's histogram slice stays in L1.
    // Every bin still receives its rows in ascending order, so the sums are
    // identical to the row sweep.
    if let Some(columns) = ghist.column_bins()
        && let Some(range) = contiguous_range(rows)
    {
        let n_rows = ghist.n_rows();
        match columns {
            Bins::U16(columns) => accumulate_columns(columns, n_rows, range, values, out),
            Bins::U32(columns) => accumulate_columns(columns, n_rows, range, values, out),
        }
        return;
    }
    if let Some(stride) = ghist.dense_stride() {
        accumulate_dense(ghist, bins, stride, rows, values, out, add_row);
    } else {
        let rp = ghist.row_ptr();
        for (i, &r) in rows.iter().enumerate() {
            if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
                let ahead = ahead as usize;
                if let (Some(&start), Some(&end)) = (rp.get(ahead), rp.get(ahead + 1)) {
                    prefetch_bins(bins, start, end - start);
                }
                if let Some(v) = values.get(ahead) {
                    crate::simd::prefetch_read(v);
                }
            }
            let ri = r as usize;
            add_row(&bins[rp[ri]..rp[ri + 1]], values[ri], out);
        }
    }
}

/// `Some(first..end)` when `rows` is exactly the ascending run
/// `first, first + 1, ..., end - 1` (checked element by element, so unsorted
/// or repeated indices never take the column path).
#[inline]
pub(super) fn contiguous_range(rows: &[u32]) -> Option<Range<usize>> {
    let first = *rows.first()? as usize;
    let end = first.checked_add(rows.len())?;
    let contiguous = rows
        .iter()
        .enumerate()
        .all(|(i, &row)| row as usize == first + i);
    contiguous.then_some(first..end)
}

/// Column-wise accumulation of the rows in `range` over every feature, four
/// features per pass so each pass reads and widens the values once for all
/// of them. `columns` is the column-major bin copy (`n_rows` entries per
/// feature). Each bin still receives its rows in ascending order.
#[inline(always)]
fn accumulate_columns<E: Bucket, V: RowValue<E>, B: BinIndex>(
    columns: &[B],
    n_rows: usize,
    range: Range<usize>,
    values: &[V],
    out: &mut [E],
) {
    fn column<'a, B>(group: &'a [B], k: usize, n_rows: usize, range: &Range<usize>) -> &'a [B] {
        &group[k * n_rows..][..n_rows][range.clone()]
    }
    let values = &values[range.clone()];
    let mut quads = columns.chunks_exact(4 * n_rows);
    for quad in &mut quads {
        let c = |k| column(quad, k, n_rows, &range);
        sweep_columns([c(0), c(1), c(2), c(3)], values, out);
    }
    let rest = quads.remainder();
    let c = |k| column(rest, k, n_rows, &range);
    match rest.len() / n_rows {
        3 => sweep_columns([c(0), c(1), c(2)], values, out),
        2 => sweep_columns([c(0), c(1)], values, out),
        1 => sweep_columns([c(0)], values, out),
        _ => {}
    }
}

/// Add each row's value to its bin in each of the `K` feature columns
/// `columns` (bins are global histogram indices, rows aligned with `values`).
/// The `K` bins of a row belong to different features, so they are distinct:
/// all are loaded before any is stored, which lets the loads overlap.
#[inline(always)]
fn sweep_columns<const K: usize, E: Bucket, V: RowValue<E>, B: BinIndex>(
    columns: [&[B]; K],
    values: &[V],
    out: &mut [E],
) {
    let n = values.len();
    let columns = columns.map(|c| &c[..n]);
    let base = out.as_mut_ptr();
    for (r, v) in values.iter().enumerate() {
        let v = v.value();
        // SAFETY: `r < n` and every column holds `n` entries.
        let bins = columns.map(|c| unsafe { c.get_unchecked(r) }.index());
        // SAFETY: every bin is `< ghist.total_bins() == out.len()` by the
        // index invariant and the caller's assertion, and the `K` bins are
        // distinct (different features of one row), so the reads and writes
        // are in bounds and never overlap.
        unsafe {
            let mut entries = bins.map(|b| *base.add(b));
            for e in &mut entries {
                e.push(v);
            }
            for (&b, e) in bins.iter().zip(entries) {
                *base.add(b) = e;
            }
        }
    }
}

/// Dense accumulation tiled by rows and feature blocks. Every bin still
/// receives its rows in ascending order, so the result is identical to a
/// straight row sweep. The tiling only changes which histogram bins are hot.
#[inline(always)]
fn accumulate_dense<E: Bucket, V: RowValue<E>, B: BinIndex>(
    ghist: &GHistIndex,
    bins: &[B],
    stride: usize,
    rows: &[u32],
    values: &[V],
    out: &mut [E],
    add_row: impl Fn(&[B], V, &mut [E]),
) {
    let blocks = feature_blocks(ghist, stride, BLOCK_BYTES / std::mem::size_of::<E>());
    // A row of at most a cache line spans one or two lines: its first and
    // last bins cover both, without a loop. Longer rows take every line.
    let short_rows = stride * std::mem::size_of::<B>() <= CACHE_LINE;
    let prefetch = |rows: &[u32], i: usize| {
        if let Some(&ahead) = rows.get(i + PREFETCH_ROWS) {
            let ahead = ahead as usize;
            if short_rows {
                if let Some(row) = bins.get(ahead * stride..(ahead + 1) * stride)
                    && let (Some(first), Some(last)) = (row.first(), row.last())
                {
                    crate::simd::prefetch_read(first);
                    crate::simd::prefetch_read(last);
                }
            } else {
                prefetch_bins(bins, ahead * stride, stride);
            }
            if let Some(v) = values.get(ahead) {
                crate::simd::prefetch_read(v);
            }
        }
    };
    if blocks.len() == 1 {
        // Dense rows sit at `r * stride`: no row-pointer load, and the address
        // of a future row is known without touching memory.
        for (i, &r) in rows.iter().enumerate() {
            prefetch(rows, i);
            let start = r as usize * stride;
            add_row(&bins[start..start + stride], values[r as usize], out);
        }
        return;
    }
    for tile in rows.chunks(TILE_ROWS) {
        for (block, &(f0, f1)) in blocks.iter().enumerate() {
            for (i, &r) in tile.iter().enumerate() {
                if block == 0 {
                    prefetch(tile, i);
                }
                let start = r as usize * stride;
                add_row(&bins[start + f0..start + f1], values[r as usize], out);
            }
        }
    }
}

/// The rows a feature-parallel build visits: a contiguous range, or an
/// ascending subset.
pub(super) enum SweepRows<'a> {
    Range(Range<usize>),
    Subset(&'a [u32]),
}

impl SweepRows<'_> {
    fn len(&self) -> usize {
        match self {
            SweepRows::Range(range) => range.len(),
            SweepRows::Subset(rows) => rows.len(),
        }
    }

    /// Rows `start..end` of the listing.
    fn part(&self, start: usize, end: usize) -> SweepRows<'_> {
        match self {
            SweepRows::Range(range) => SweepRows::Range(range.start + start..range.start + end),
            SweepRows::Subset(rows) => SweepRows::Subset(&rows[start..end]),
        }
    }
}

/// Build `out` (overwritten) over `rows` from the column-major copy
/// `columns`, split by feature instead of by rows: each task streams its
/// features' columns straight into their slices of `out`. The sums follow
/// the blocked order (see [`super::SumOrder::Blocked`]): every
/// `grain`-row chunk is chained from zero into a per-feature partial, and
/// the partials are added in chunk order, the first copied, which is the
/// row-parallel blocked build's result bit for bit. Tasks take as few
/// features as keep every one of `threads` workers busy (at most four),
/// swept together so each pass reads and widens the row values once for
/// all of them.
pub(super) fn by_features<E: Bucket, V: RowValue<E>>(
    ghist: &GHistIndex,
    columns: &Bins<'_>,
    rows: &SweepRows<'_>,
    values: &[V],
    out: &mut [E],
    grain: usize,
    threads: usize,
) {
    let n_rows = ghist.n_rows();
    let per_task = ghist.n_cols().div_ceil(threads).clamp(1, 4);
    let mut slices = feature_slices(ghist, out, 1);
    slices
        .par_chunks_mut(per_task)
        .enumerate()
        .for_each(|(task, features)| {
            let f = task * per_task;
            match columns {
                Bins::U16(c) => feature_group(c, n_rows, f, features, rows, values, grain),
                Bins::U32(c) => feature_group(c, n_rows, f, features, rows, values, grain),
            }
        });
}

/// Blocked accumulation of `rows` for the (up to four) consecutive
/// features `f..` whose histogram slices `features` holds (`features[k]` is
/// feature `f + k`'s first global bin and its slice, overwritten). Each
/// chunk's sweep reads each row's value once for all the features. Bins in
/// a column are global indices, so each slice is indexed relative to its
/// first bin; the binned-index invariant keeps every bin of a feature's
/// column in its range, and the slice index bounds check catches any
/// violation.
#[inline(always)]
fn feature_group<E: Bucket, V: RowValue<E>, B: BinIndex>(
    columns: &[B],
    n_rows: usize,
    f: usize,
    features: &mut [(usize, &mut [E])],
    rows: &SweepRows<'_>,
    values: &[V],
    grain: usize,
) {
    let column = |k: usize| &columns[(f + k) * n_rows..][..n_rows];
    // One partial per feature, reused by every chunk.
    let mut partials: Vec<(usize, Vec<E>)> = features
        .iter()
        .map(|(first, slice)| (*first, vec![E::default(); slice.len()]))
        .collect();
    let n = rows.len();
    for (c, start) in (0..n).step_by(grain.max(1)).enumerate() {
        let chunk = rows.part(start, (start + grain).min(n));
        for (_, partial) in &mut partials {
            partial.fill(E::default());
        }
        match partials.as_mut_slice() {
            [a] => sweep_slices([column(0)], [a], &chunk, values),
            [a, b] => sweep_slices([column(0), column(1)], [a, b], &chunk, values),
            [a, b, cc] => {
                sweep_slices(
                    [column(0), column(1), column(2)],
                    [a, b, cc],
                    &chunk,
                    values,
                );
            }
            [a, b, cc, d] => sweep_slices(
                [column(0), column(1), column(2), column(3)],
                [a, b, cc, d],
                &chunk,
                values,
            ),
            _ => unreachable!("callers pass one to four features"),
        }
        for ((_, slice), (_, partial)) in features.iter_mut().zip(&partials) {
            if c == 0 {
                slice.copy_from_slice(partial);
            } else {
                for (o, &p) in slice.iter_mut().zip(partial.iter()) {
                    o.push(p);
                }
            }
        }
    }
}

/// Add each row's value to its bin in each of the `K` full-length feature
/// columns, into that feature's `(first bin, partial)`.
#[inline(always)]
fn sweep_slices<const K: usize, E: Bucket, V: RowValue<E>, B: BinIndex>(
    columns: [&[B]; K],
    slices: [&mut (usize, Vec<E>); K],
    rows: &SweepRows<'_>,
    values: &[V],
) {
    let mut slices = slices.map(|(first, slice)| (*first, slice.as_mut_slice()));
    match rows {
        SweepRows::Range(range) => {
            let values = &values[range.clone()];
            let n = values.len();
            let columns = columns.map(|c| &c[range.clone()][..n]);
            for (r, v) in values.iter().enumerate() {
                let v = v.value();
                for (column, (first, slice)) in columns.iter().zip(&mut slices) {
                    slice[column[r].index() - *first].push(v);
                }
            }
        }
        SweepRows::Subset(rows) => {
            for &r in *rows {
                let r = r as usize;
                let v = values[r].value();
                for (column, (first, slice)) in columns.iter().zip(&mut slices) {
                    slice[column[r].index() - *first].push(v);
                }
            }
        }
    }
}
