//! Runtime-dispatched numeric kernels for operations the compiler cannot
//! auto-vectorize, chiefly transcendental objective functions.

mod scalar;

const LOG_LOSS_EPSILON: f64 = 1e-15;
/// XGBoost's binary `logloss` floor (`float eps = 1e-16`), widened to `f64`.
const BINARY_LOG_LOSS_EPSILON: f64 = 1e-16f32 as f64;
const MIN_POSITIVE_PREDICTION: f64 = 1e-8;

use crate::objective::GradPair;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use std::sync::LazyLock;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const MIN_SIMD_LEN: usize = 16;
// `objective::GRADIENT_BLOCK_ROWS`'s contract: a block that long takes the
// vector path.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const _: () = assert!(MIN_SIMD_LEN <= crate::objective::GRADIENT_BLOCK_ROWS);

// Layout contract the deinterleaving vector loads and stores rely on.
const _: () = assert!(std::mem::size_of::<GradPair>() == 2 * std::mem::size_of::<f32>());

/// Inputs with a larger magnitude take the scalar path in the fast
/// exponential, sigmoid, and softmax kernels.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const MAX_FAST_EXP_INPUT: f32 = 80.0;

/// How [`logistic_gradient`] runs one whole batch of rows on this machine,
/// for a device that reproduces its vector kernel: `lanes` rows per vector
/// over the batch's first `rows` rows, the rest scalar. A vector holding a
/// margin of magnitude above `max_input` (or a non-finite one) also runs
/// scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorSplit {
    pub lanes: usize,
    pub rows: usize,
    pub max_input: f32,
}

/// [`VectorSplit`] of a batch of `n` rows; `None` when the logistic
/// gradient runs scalar everywhere (no vector kernel on this machine).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) fn logistic_vector_split(n: usize) -> Option<VectorSplit> {
    #[cfg(target_arch = "aarch64")]
    let lanes = neon_available().then_some(aarch64::VECTOR_WIDTH);
    #[cfg(target_arch = "x86_64")]
    let lanes = avx2_fma_available().then_some(x86_64::WIDTH);
    lanes.map(|lanes| VectorSplit {
        lanes,
        rows: if n >= MIN_SIMD_LEN { n - n % lanes } else { 0 },
        max_input: MAX_FAST_EXP_INPUT,
    })
}

/// [`VectorSplit`] of a batch of `n` rows; `None` when the logistic
/// gradient runs scalar everywhere (no vector kernel on this machine).
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn logistic_vector_split(_n: usize) -> Option<VectorSplit> {
    None
}

#[cfg(target_arch = "aarch64")]
static NEON_AVAILABLE: LazyLock<bool> =
    LazyLock::new(|| std::arch::is_aarch64_feature_detected!("neon"));

#[cfg(target_arch = "x86_64")]
static AVX2_FMA_AVAILABLE: LazyLock<bool> = LazyLock::new(|| {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
});

/// Run the per-arch kernel for a `&mut [f32]` unary inplace op when the slice
/// is long enough, falling through to the caller's scalar tail otherwise.
macro_rules! dispatch_unary_inplace {
    ($values:expr, $kernel:ident) => {
        #[cfg(target_arch = "aarch64")]
        if $values.len() >= MIN_SIMD_LEN && neon_available() {
            // SAFETY: runtime detection proves NEON is present and the kernel
            // bounds vector accesses by the slice length.
            unsafe { aarch64::$kernel($values) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if $values.len() >= MIN_SIMD_LEN && avx2_fma_available() {
            // SAFETY: AVX2/FMA are present and the kernel bounds vector
            // accesses by the slice length.
            unsafe { x86_64::$kernel($values) };
            return;
        }
    };
}

/// Run the per-arch gradient kernel when `$gate` (typically `gradient_gate`
/// or `metric_gate`) holds, falling through to the caller's scalar
/// tail otherwise. The three-arm form also dispatches the `x86_64` kernel; the
/// two-arm form is NEON-only for kernels with no `x86_64` counterpart. Both
/// forms return the kernel's value, so `()`-valued gradient kernels and
/// `(f64, f64)`-valued metric-sum kernels share the same expansion.
macro_rules! dispatch_gradient {
    ($gate:expr, $neon_call:expr) => {
        #[cfg(target_arch = "aarch64")]
        if $gate && neon_available() {
            // SAFETY: NEON is present and the gate's cover check proves every
            // input and output slice spans the dispatched length.
            return unsafe { $neon_call };
        }
    };
    ($gate:expr, $neon_call:expr, $avx_call:expr) => {
        dispatch_gradient!($gate, $neon_call);
        #[cfg(target_arch = "x86_64")]
        if $gate && avx2_fma_available() {
            // SAFETY: AVX2/FMA are present and the gate's cover check proves
            // every input and output slice spans the dispatched length.
            unsafe { $avx_call };
            return;
        }
    };
}

/// Dispatch a softmax entry point to the vector kernels: on AArch64 the
/// wide-row kernel (`$wide`) for 8+ classes and the short-row kernels for
/// 2–4 classes, on `x86_64` the short-row kernels for 2 and 4 classes, each
/// when `$eligible` holds. `$short` calls the short-row kernel as
/// `arch::kernel::<$k>(..)`, with `arch` the architecture's module and `$k`
/// the class count. Returns after a vector kernel ran; otherwise falls
/// through to the caller's scalar path.
macro_rules! dispatch_softmax {
    ($num_class:expr, $eligible:expr, wide: $wide:expr, short: |$k:ident| $short:expr) => {
        #[cfg(target_arch = "aarch64")]
        if $eligible && neon_available() {
            if $num_class >= 8 {
                // SAFETY: NEON is present; the caller's eligibility check
                // covers the complete row-major matrix, and the kernel bounds
                // each row by `num_class`.
                unsafe { $wide };
                return;
            }
            if (2..=4).contains(&$num_class) {
                // SAFETY: NEON is present, the eligibility check covers the
                // matrix, and each specialization processes bounded batches of
                // four rows and handles the remaining values scalarly.
                unsafe {
                    match $num_class {
                        2 => {
                            use aarch64 as arch;
                            const $k: usize = 2;
                            $short
                        }
                        3 => {
                            use aarch64 as arch;
                            const $k: usize = 3;
                            $short
                        }
                        _ => {
                            use aarch64 as arch;
                            const $k: usize = 4;
                            $short
                        }
                    }
                }
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        if ($num_class == 2 || $num_class == 4) && $eligible && avx2_fma_available() {
            // SAFETY: AVX2/FMA are present, the eligibility check covers the
            // matrix, and each specialization processes whole rows per vector
            // and handles the remaining rows scalarly.
            unsafe {
                if $num_class == 2 {
                    use x86_64 as arch;
                    const $k: usize = 2;
                    $short
                } else {
                    use x86_64 as arch;
                    const $k: usize = 4;
                    $short
                }
            }
            return;
        }
    };
}

/// Resolve the process-wide AArch64 backend lazily on the first numeric-kernel
/// call. `LazyLock` makes feature detection a one-time initialization cost, and
/// subsequent calls are a cached load and comparison. A build whose target
/// enables NEON (every standard AArch64 target) needs no check at all, which
/// keeps per-element callers (SHAP's edge terms, cut search) free of the
/// load's acquire ordering.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_available() -> bool {
    cfg!(target_feature = "neon") || *NEON_AVAILABLE
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_fma_available() -> bool {
    *AVX2_FMA_AVAILABLE
}

/// Hint the cache hierarchy that `value` will be read soon. A pure performance
/// hint: it never faults and has no observable effect on program state.
#[inline(always)]
pub(crate) fn prefetch_read<T>(value: &T) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: PRFM only touches the cache and cannot fault or write memory;
    // the operand is a valid reference.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) std::ptr::from_ref::<T>(value),
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: PREFETCHT0 is available on every x86_64 CPU and cannot fault.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
            std::ptr::from_ref::<T>(value).cast::<i8>(),
        );
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = value;
}

/// `base + 1` when `a > b`, else `base`, computed without a branch.
///
/// Tree walks step to `left + (key > threshold)` for rows whose outcomes
/// are data-dependent and unpredictable; LLVM's AArch64 backend lowers that
/// select (in any spelling) to a conditional branch, which such rows
/// mispredict about a quarter of the time. The AArch64 path pins the
/// conditional increment in one instruction.
#[inline(always)]
pub(crate) fn step_if_greater(base: usize, a: u32, b: u32) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        let out: usize;
        // SAFETY: a compare and a conditional increment of registers; no
        // memory access, and `preserves_flags` is not claimed, so the
        // compiler treats the condition flags as clobbered.
        unsafe {
            std::arch::asm!(
                "cmp {a:w}, {b:w}",
                "cinc {out}, {base}, hi",
                a = in(reg) a,
                b = in(reg) b,
                base = in(reg) base,
                out = lateout(reg) out,
                options(pure, nomem, nostack)
            );
        }
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        base + usize::from(a > b)
    }
}

/// Number of entries of `cuts` that are `<= value` (ordered comparison).
#[inline]
pub(crate) fn count_le(cuts: &[f32], value: f32) -> usize {
    #[cfg(target_arch = "aarch64")]
    if cuts.len() == 16 && neon_available() {
        // SAFETY: NEON is present and the slice holds exactly four vectors.
        return unsafe { aarch64::count_le_16(cuts, value) };
    }
    #[cfg(target_arch = "x86_64")]
    if cuts.len() == 16 {
        // SAFETY: the slice holds exactly four vectors; SSE2 is baseline.
        return unsafe { x86_64::count_le_16(cuts, value) };
    }
    cuts.iter().filter(|&&cut| cut <= value).count()
}

/// QuadratureTreeSHAP's return-edge terms `α·h[i] / (1 + α·u[i])` for the
/// eight quadrature lanes, with the denominator's multiply-add fused on
/// AArch64 (as XGBoost's builds there contract it) and unfused elsewhere.
/// Every lane is the scalar expression's value exactly.
#[inline(always)]
pub(crate) fn shap_edge_terms(alpha: f32, h: &[f32; 8], u: &[f32; 8]) -> [f32; 8] {
    #[cfg(target_arch = "aarch64")]
    if neon_available() {
        // SAFETY: NEON is present.
        return unsafe { aarch64::edge_terms_8(alpha, h, u) };
    }
    std::array::from_fn(|i| alpha * h[i] / shap_denominator(alpha, u[i]))
}

/// QuadratureTreeSHAP's child basis `c[i] · (1 + α·u[i])` for the eight
/// quadrature lanes, the multiply-add fused as in [`shap_edge_terms`]. Every
/// lane is the scalar expression's value exactly.
#[inline(always)]
pub(crate) fn shap_scaled_basis(alpha: f32, c: &[f32; 8], u: &[f32; 8]) -> [f32; 8] {
    #[cfg(target_arch = "aarch64")]
    if neon_available() {
        // SAFETY: NEON is present.
        return unsafe { aarch64::scaled_basis_8(alpha, c, u) };
    }
    std::array::from_fn(|i| c[i] * shap_denominator(alpha, u[i]))
}

/// `c[i] / (1 + α·u[i])` for the eight quadrature lanes (the multiply-add
/// fused as in [`shap_edge_terms`]) when every `c[i]` and every denominator
/// is finite, else `None`. Every lane is the scalar expression's value
/// exactly.
#[inline(always)]
pub(crate) fn shap_divided_basis(alpha: f32, c: &[f32; 8], u: &[f32; 8]) -> Option<[f32; 8]> {
    #[cfg(target_arch = "aarch64")]
    if neon_available() {
        // SAFETY: NEON is present.
        return unsafe { aarch64::divided_basis_8(alpha, c, u) };
    }
    let old: [f32; 8] = std::array::from_fn(|i| shap_denominator(alpha, u[i]));
    c.iter()
        .zip(&old)
        .all(|(c, o)| c.is_finite() && o.is_finite())
        .then(|| std::array::from_fn(|i| c[i] / old[i]))
}

/// `1 + α·u`, fused on AArch64 (as XGBoost's builds there contract it).
#[inline(always)]
fn shap_denominator(alpha: f32, u: f32) -> f32 {
    if cfg!(target_arch = "aarch64") {
        alpha.mul_add(u, 1.0)
    } else {
        alpha * u + 1.0
    }
}

/// Whether a gradient kernel may take the vector path: at least
/// `MIN_SIMD_LEN` predictions, with labels, weights, and `out` covering them.
#[inline]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn gradient_gate(preds: &[f32], labels: &[f32], weights: Option<&[f32]>, out: &[GradPair]) -> bool {
    metric_gate(preds, labels, weights.map(RowWeights::from)) && out.len() >= preds.len()
}

/// Whether a metric-sum kernel may take the vector path: at least
/// `MIN_SIMD_LEN` predictions, with labels and weights covering them.
#[inline]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn metric_gate(preds: &[f32], labels: &[f32], weights: Option<RowWeights<'_>>) -> bool {
    let len = preds.len();
    len >= MIN_SIMD_LEN
        && labels.len() >= len
        && weights.is_none_or(|weights| weights.cells().is_some_and(|cells| cells >= len))
}

/// Row weights of a `[row][target]` cell layout: cell `i` has weight
/// `values[i / stride]`, so the metric sums read each row's weight for its
/// `stride` cells without materializing the repeated weights. A stride of
/// one is one weight per cell.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowWeights<'a> {
    values: &'a [f32],
    stride: usize,
}

impl<'a> RowWeights<'a> {
    /// `values` repeated for `stride` (positive) consecutive cells each.
    pub(crate) fn new(values: &'a [f32], stride: usize) -> Self {
        debug_assert!(stride > 0, "row weight stride must be positive");
        RowWeights { values, stride }
    }

    /// The weight of cell `cell`.
    #[inline]
    pub(crate) fn get(self, cell: usize) -> f32 {
        if self.stride == 1 {
            self.values[cell]
        } else {
            self.values[cell / self.stride]
        }
    }

    /// The number of cells the weights cover, `None` on overflow.
    #[inline]
    pub(crate) fn cells(self) -> Option<usize> {
        self.values.len().checked_mul(self.stride)
    }
}

impl<'a> From<&'a [f32]> for RowWeights<'a> {
    /// One weight per cell.
    fn from(values: &'a [f32]) -> Self {
        Self::new(values, 1)
    }
}

/// Whether `labels` (and `weights`, if any) index complete `num_class` rows
/// of `preds`.
#[inline]
fn class_rows_cover(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> bool {
    labels
        .len()
        .checked_mul(num_class)
        .is_some_and(|len| preds.len() >= len)
        && weights.is_none_or(|values| values.len() >= labels.len())
}

/// XGBoost's `common::Sigmoid`: `1 / (expf(min(-x, 88.7)) + 1)` (the
/// `1e-16f` upstream adds to the denominator vanishes in `f32`).
#[inline]
pub(crate) fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / ((-x).min(88.7).exp() + 1.0)
}

#[inline]
pub(crate) fn exp_inplace(values: &mut [f32]) {
    dispatch_unary_inplace!(values, exp_inplace);
    for value in values.iter_mut() {
        *value = value.exp();
    }
}

#[inline]
pub(crate) fn sigmoid_inplace(values: &mut [f32]) {
    dispatch_unary_inplace!(values, sigmoid_inplace);
    for value in values.iter_mut() {
        *value = sigmoid_scalar(*value);
    }
}

pub(crate) fn logistic_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    dispatch_gradient!(
        gradient_gate(preds, labels, weights, out),
        aarch64::logistic_gradient(preds, labels, weights, scale_pos_weight, min_hess, out),
        x86_64::logistic_gradient(preds, labels, weights, scale_pos_weight, min_hess, out)
    );
    scalar::logistic_gradient(
        preds,
        labels,
        weights,
        scale_pos_weight,
        min_hess,
        out,
        0..preds.len(),
    );
}

pub(crate) fn poisson_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    max_delta_step: f32,
    out: &mut [GradPair],
) {
    dispatch_gradient!(
        gradient_gate(preds, labels, weights, out),
        aarch64::poisson_gradient(preds, labels, weights, max_delta_step, out)
    );
    scalar::poisson_gradient(preds, labels, weights, max_delta_step, out, 0..preds.len());
}

pub(crate) fn gamma_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    out: &mut [GradPair],
) {
    dispatch_gradient!(
        gradient_gate(preds, labels, weights, out),
        aarch64::gamma_gradient(preds, labels, weights, scale_pos_weight, out)
    );
    scalar::gamma_gradient(
        preds,
        labels,
        weights,
        scale_pos_weight,
        out,
        0..preds.len(),
    );
}

pub(crate) fn tweedie_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f32,
    out: &mut [GradPair],
) {
    dispatch_gradient!(
        gradient_gate(preds, labels, weights, out),
        aarch64::tweedie_gradient(preds, labels, weights, rho, out)
    );
    scalar::tweedie_gradient(preds, labels, weights, rho, out, 0..preds.len());
}

/// Apply softmax to every contiguous `num_class` row while resolving the SIMD
/// backend only once for the whole matrix.
pub(crate) fn softmax_rows_inplace(values: &mut [f32], num_class: usize) {
    dispatch_softmax!(
        num_class,
        values.len() >= MIN_SIMD_LEN,
        wide: aarch64::softmax_rows_inplace(values, num_class),
        short: |K| arch::short_softmax_rows::<K>(values)
    );

    for row in values.chunks_mut(num_class) {
        softmax_scalar(row);
    }
}

pub(crate) fn softmax_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
    min_hess: f32,
    out: &mut [GradPair],
) {
    let complete = labels
        .len()
        .checked_mul(num_class)
        .is_some_and(|len| len == preds.len() && out.len() >= len)
        && weights.is_none_or(|values| values.len() >= labels.len());
    dispatch_softmax!(
        num_class,
        preds.len() >= MIN_SIMD_LEN && complete,
        wide: aarch64::softmax_gradient(preds, labels, weights, num_class, min_hess, out),
        short: |K| arch::short_softmax_gradient::<K>(preds, labels, weights, min_hess, out)
    );

    debug_assert!(complete);
    softmax_gradient_rows_scalar(
        preds,
        labels,
        weights,
        min_hess,
        out,
        0..labels.len(),
        num_class,
    );
}

/// Scalar softmax gradient over the given rows of a complete row-major
/// prediction matrix; the remainder path of the short-row vector kernels.
pub(super) fn softmax_gradient_rows_scalar(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    min_hess: f32,
    out: &mut [GradPair],
    rows: std::ops::Range<usize>,
    k: usize,
) {
    for current in rows {
        let base = current * k;
        softmax_gradient_row_scalar(
            &preds[base..base + k],
            labels[current] as usize,
            weights.map_or(1.0, |values| values[current]),
            min_hess,
            &mut out[base..base + k],
        );
    }
}

/// XGBoost's `common::Softmax`: shift by the row maximum, sum the
/// exponentials in `f64`, and divide each entry by that sum rounded to `f32`.
pub(super) fn softmax_scalar(values: &mut [f32]) {
    let Some(&first) = values.first() else {
        return;
    };
    let wmax = values[1..].iter().fold(first, |m, &v| v.max(m));
    let mut wsum = 0f64;
    for value in values.iter_mut() {
        *value = (*value - wmax).exp();
        wsum += f64::from(*value);
    }
    let wsum = wsum as f32;
    for value in values.iter_mut() {
        *value /= wsum;
    }
}

/// XGBoost's `SoftmaxMultiClassObj::GetGradient` for one row: the shift is
/// `max(f32::MIN_POSITIVE, preds...)` (upstream seeds `wmax` with
/// `numeric_limits<float>::min()`), the exponentials are summed in `f64`, and
/// `p = expf(x - wmax) / (float)wsum`.
pub(super) fn softmax_gradient_row_scalar(
    preds: &[f32],
    label: usize,
    weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    let mut wmax = f32::MIN_POSITIVE;
    for &value in preds {
        wmax = value.max(wmax);
    }
    let mut wsum = 0f64;
    for &prediction in preds {
        wsum += f64::from((prediction - wmax).exp());
    }
    let wsum = wsum as f32;
    for (class, (output, &prediction)) in out.iter_mut().zip(preds).enumerate() {
        let p = (prediction - wmax).exp() / wsum;
        let h = (2.0 * p * (1.0 - p) * weight).max(min_hess);
        let g = if class == label { p - 1.0 } else { p };
        *output = GradPair::new(g * weight, h);
    }
}

pub(crate) fn squared_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    distance_sum::<true>(preds, labels, weights)
}

pub(crate) fn absolute_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    distance_sum::<false>(preds, labels, weights)
}

fn distance_sum<const SQUARED: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    dispatch_gradient!(
        metric_gate(preds, labels, weights),
        aarch64::distance_sum::<SQUARED>(preds, labels, weights)
    );

    scalar::distance_sum::<SQUARED>(preds, labels, weights, 0..preds.len())
}

pub(crate) fn classification_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    dispatch_gradient!(
        metric_gate(preds, labels, weights),
        aarch64::classification_error_sum(preds, labels, weights)
    );

    scalar::classification_error_sum(preds, labels, weights, 0..preds.len())
}

pub(crate) fn log_loss_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    dispatch_gradient!(
        metric_gate(preds, labels, weights),
        aarch64::log_loss_sum(preds, labels, weights)
    );

    scalar::log_loss(preds, labels, weights, 0..preds.len())
}

pub(crate) fn positive_nloglik_sum<const GAMMA: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    dispatch_gradient!(
        metric_gate(preds, labels, weights),
        aarch64::positive_nloglik_sum::<GAMMA>(preds, labels, weights)
    );

    scalar::positive_nloglik::<GAMMA>(preds, labels, weights, 0..preds.len())
}

pub(crate) fn tweedie_nloglik_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
    rho: f64,
) -> (f64, f64) {
    dispatch_gradient!(
        rho.is_finite() && rho > 1.0 && rho < 2.0 && metric_gate(preds, labels, weights),
        aarch64::tweedie_nloglik_sum(preds, labels, weights, rho)
    );

    scalar::tweedie_nloglik(preds, labels, weights, rho, 0..preds.len())
}

pub(crate) fn multiclass_log_loss_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    let complete = class_rows_cover(preds, labels, weights, num_class);
    dispatch_gradient!(
        labels.len() >= MIN_SIMD_LEN && complete,
        aarch64::multiclass_log_loss_sum(preds, labels, weights, num_class)
    );

    debug_assert!(complete);
    scalar::multiclass_log_loss(preds, labels, weights, num_class, 0..labels.len())
}

pub(crate) fn multiclass_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    let complete = class_rows_cover(preds, labels, weights, num_class);
    dispatch_gradient!(
        num_class >= 8
            && u32::try_from(num_class).is_ok()
            && labels.len() >= MIN_SIMD_LEN
            && complete,
        aarch64::multiclass_error_sum(preds, labels, weights, num_class)
    );

    debug_assert!(complete);
    multiclass_error_sum_rows(preds, labels, weights, num_class, argmax_scalar)
}

/// Row loop of the multiclass error sum, parameterized on the argmax so the
/// NEON kernel can reuse it with its vectorized `argmax`.
pub(super) fn multiclass_error_sum_rows(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
    argmax: impl Fn(&[f32]) -> usize,
) -> (f64, f64) {
    let mut wrong = 0.0;
    let mut weight_sum = 0.0;
    for (row_index, &label) in labels.iter().enumerate() {
        let weight = weights.map_or(1.0, |values| f64::from(values[row_index]));
        let row = &preds[row_index * num_class..(row_index + 1) * num_class];
        let best = argmax(row);
        if best != label as usize {
            wrong += weight;
        }
        weight_sum += weight;
    }
    (wrong, weight_sum)
}

pub(crate) fn argmax_scalar(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests;
