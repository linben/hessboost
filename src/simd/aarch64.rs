use super::{
    BINARY_LOG_LOSS_EPSILON, LOG_LOSS_EPSILON, MAX_FAST_EXP_INPUT, MIN_POSITIVE_PREDICTION,
    RowWeights, scalar, sigmoid_scalar,
};
use crate::objective::GradPair;
#[allow(
    clippy::wildcard_imports,
    reason = "intrinsic modules are used wholesale"
)]
use std::arch::aarch64::*;

// Arithmetic-only helpers express the NEON precondition through an unsafe
// intrinsic function pointer. This keeps their unsafe blocks valid whether or
// not the stdarch in use marks these intrinsics safe, without lint overrides.
// The compiler inlines these constant function pointers.

pub(super) const VECTOR_WIDTH: usize = 4;
// `objective::GRADIENT_BLOCK_ROWS`'s contract: its blocks start on vector blocks.
const _: () = assert!(crate::objective::GRADIENT_BLOCK_ROWS.is_multiple_of(VECTOR_WIDTH));
// Shared vector-loop scaffolding for the gradient and metric-sum kernels below.
// Lane formulas stay inline in each kernel; only the identical accumulate,
// fallback, store, and reduction shells live here, so numerics are untouched.
macro_rules! accumulate_metric_sum {
    ($weights:expr, $index:ident, $value_low:expr, $value_high:expr, $sum_low:ident, $sum_high:ident, $weight_low:ident, $weight_high:ident) => {
        match $weights {
            Some(weights) => {
                let weight = lane_weights(RowWeights::from(weights), $index);
                let current_weight_low = vcvt_f64_f32(vget_low_f32(weight));
                let current_weight_high = vcvt_high_f64_f32(weight);
                $sum_low = vfmaq_f64($sum_low, $value_low, current_weight_low);
                $sum_high = vfmaq_f64($sum_high, $value_high, current_weight_high);
                $weight_low = vaddq_f64($weight_low, current_weight_low);
                $weight_high = vaddq_f64($weight_high, current_weight_high);
            }
            None => {
                $sum_low = vaddq_f64($sum_low, $value_low);
                $sum_high = vaddq_f64($sum_high, $value_high);
            }
        }
    };
}
macro_rules! metric_finite_guard {
    ($pred:expr, $index:ident, $fallback:ident, $scalar:expr) => {
        if !finite_input($pred) {
            let partial = $scalar;
            $fallback.0 += partial.0;
            $fallback.1 += partial.1;
            $index += VECTOR_WIDTH;
            continue;
        }
    };
}
macro_rules! finish_metric_sum {
    ($sum_low:ident, $sum_high:ident, $weight_low:ident, $weight_high:ident, $fallback:expr, $tail:expr, $weights:expr, $len:expr) => {{
        let loss = vaddvq_f64(vaddq_f64($sum_low, $sum_high)) + $fallback.0 + $tail.0;
        let weight_sum = match $weights {
            Some(_) => vaddvq_f64(vaddq_f64($weight_low, $weight_high)) + $fallback.1 + $tail.1,
            None => $len as f64,
        };
        (loss, weight_sum)
    }};
}
macro_rules! gradient_guard {
    ($cond:expr, $index:ident, $fallback:expr) => {
        if $cond {
            $fallback;
            $index += VECTOR_WIDTH;
            continue;
        }
    };
}
/// The weights of the four cells from `index`: one vector load for one
/// weight per cell, else each cell's row weight gathered into the lanes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn lane_weights(weights: RowWeights<'_>, index: usize) -> float32x4_t {
    if weights.stride == 1 {
        // SAFETY: the caller guarantees NEON support and weights covering
        // the four cells from `index`.
        unsafe { vld1q_f32(weights.values.as_ptr().add(index)) }
    } else {
        let lanes = [
            weights.get(index),
            weights.get(index + 1),
            weights.get(index + 2),
            weights.get(index + 3),
        ];
        // SAFETY: the caller guarantees NEON support; `lanes` holds four f32s.
        unsafe { vld1q_f32(lanes.as_ptr()) }
    }
}
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_grad_pairs(out: *mut GradPair, index: usize, grad: float32x4_t, hess: float32x4_t) {
    // SAFETY: the caller guarantees NEON support and four writable pairs at `index`.
    unsafe {
        vst2q_f32(out.add(index).cast::<f32>(), float32x4x2_t(grad, hess));
    }
}

/// Exponential for finite f32 lanes in [-80, 80]. Range
/// reduction keeps the polynomial input in [-ln(2)/2, ln(2)/2], where a
/// seventh-order Taylor polynomial is within a few f32 ULPs.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn expq_f32<const ESTRIN: bool>(value: float32x4_t) -> float32x4_t {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let scaled = vmulq_n_f32(value, std::f32::consts::LOG2_E);
        let exponent = vcvtnq_s32_f32(scaled);
        let exponent_f32 = vcvtq_f32_s32(exponent);

        // Split ln(2) so the range reduction loses fewer low bits.
        let mut reduced = vfmsq_n_f32(value, exponent_f32, 0.693_359_4);
        reduced = vfmaq_n_f32(reduced, exponent_f32, 2.121_944_4e-4);

        let polynomial = if ESTRIN {
            // Estrin evaluation exposes independent pairs instead of seven dependent FMAs.
            let squared = vmulq_f32(reduced, reduced);
            let fourth = vmulq_f32(squared, squared);
            let pair_0 = vaddq_f32(vdupq_n_f32(1.0), reduced);
            let pair_1 = vfmaq_f32(vdupq_n_f32(0.5), vdupq_n_f32(1.0 / 6.0), reduced);
            let pair_2 = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), vdupq_n_f32(1.0 / 120.0), reduced);
            let pair_3 = vfmaq_f32(
                vdupq_n_f32(1.0 / 720.0),
                vdupq_n_f32(1.0 / 5_040.0),
                reduced,
            );
            let low = vfmaq_f32(pair_0, pair_1, squared);
            let high = vfmaq_f32(pair_2, pair_3, squared);
            vfmaq_f32(low, high, fourth)
        } else {
            // Wide in-place softmax favors Horner evaluation for throughput.
            let mut polynomial = vdupq_n_f32(1.0 / 5_040.0);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0 / 720.0), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(0.5), polynomial, reduced);
            polynomial = vfmaq_f32(vdupq_n_f32(1.0), polynomial, reduced);
            vfmaq_f32(vdupq_n_f32(1.0), polynomial, reduced)
        };

        let exponent_bits = vshlq_n_s32(vaddq_s32(exponent, vdupq_n_s32(127)), 23);
        let multiply: unsafe fn(float32x4_t, float32x4_t) -> float32x4_t = vmulq_f32;
        multiply(polynomial, vreinterpretq_f32_s32(exponent_bits))
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn sigmoidq_f32(value: float32x4_t) -> float32x4_t {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let magnitude = vabsq_f32(value);
        let exp = expq_f32::<true>(vnegq_f32(magnitude));
        let denominator = vaddq_f32(vdupq_n_f32(1.0), exp);
        let positive = vdivq_f32(vdupq_n_f32(1.0), denominator);
        let negative = vdivq_f32(exp, denominator);
        vbslq_f32(vcgeq_f32(value, vdupq_n_f32(0.0)), positive, negative)
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn regular_input(value: float32x4_t) -> bool {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let magnitude = vabsq_f32(value);
        let in_range = vcleq_f32(magnitude, vdupq_n_f32(MAX_FAST_EXP_INPUT));
        let minimum: unsafe fn(uint32x4_t) -> u32 = vminvq_u32;
        minimum(in_range) == u32::MAX
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn finite_input(value: float32x4_t) -> bool {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let finite = vcleq_f32(vabsq_f32(value), vdupq_n_f32(f32::MAX));
        let minimum: unsafe fn(uint32x4_t) -> u32 = vminvq_u32;
        minimum(finite) == u32::MAX
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn finite_min_max(values: &[f32]) -> Option<(f32, f32)> {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let mut maximum = vdupq_n_f32(f32::NEG_INFINITY);
        let mut minimum = vdupq_n_f32(f32::INFINITY);
        let mut index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable values.
            let value = vld1q_f32(values.as_ptr().add(index));
            // These min/max instructions propagate NaNs, so checking the final
            // extrema also rejects non-finite lanes from any earlier block.
            maximum = vmaxq_f32(maximum, value);
            minimum = vminq_f32(minimum, value);
            index += VECTOR_WIDTH;
        }

        let mut max = vmaxvq_f32(maximum);
        let mut min = vminvq_f32(minimum);
        if !min.is_finite() || !max.is_finite() {
            return None;
        }
        for &value in &values[index..] {
            if !value.is_finite() {
                return None;
            }
            max = max.max(value);
            min = min.min(value);
        }
        Some((min, max))
    }
}

/// Natural logarithm for positive, finite, normal f64 lanes. Mantissas are
/// reduced around one and evaluated with an atanh series through z^15.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn logq_f64(value: float64x2_t) -> float64x2_t {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        const MANTISSA_MASK: u64 = (1u64 << 52) - 1;
        const ONE_BITS: u64 = 1023u64 << 52;
        let bits = vreinterpretq_u64_f64(value);
        let raw_exponent = vandq_u64(vshrq_n_u64::<52>(bits), vdupq_n_u64(0x7ff));
        let mut exponent = vsubq_f64(vcvtq_f64_u64(raw_exponent), vdupq_n_f64(1023.0));
        let mut mantissa = vreinterpretq_f64_u64(vorrq_u64(
            vandq_u64(bits, vdupq_n_u64(MANTISSA_MASK)),
            vdupq_n_u64(ONE_BITS),
        ));

        let halve = vcgtq_f64(mantissa, vdupq_n_f64(std::f64::consts::SQRT_2));
        mantissa = vbslq_f64(halve, vmulq_n_f64(mantissa, 0.5), mantissa);
        exponent = vaddq_f64(
            exponent,
            vbslq_f64(halve, vdupq_n_f64(1.0), vdupq_n_f64(0.0)),
        );

        let one = vdupq_n_f64(1.0);
        let z = vdivq_f64(vsubq_f64(mantissa, one), vaddq_f64(mantissa, one));
        let z_squared = vmulq_f64(z, z);
        // Estrin evaluation exposes independent FMAs instead of a seven-FMA chain.
        let z_fourth = vmulq_f64(z_squared, z_squared);
        let z_eighth = vmulq_f64(z_fourth, z_fourth);
        let pair_0 = vfmaq_f64(one, vdupq_n_f64(1.0 / 3.0), z_squared);
        let pair_1 = vfmaq_f64(vdupq_n_f64(1.0 / 5.0), vdupq_n_f64(1.0 / 7.0), z_squared);
        let pair_2 = vfmaq_f64(vdupq_n_f64(1.0 / 9.0), vdupq_n_f64(1.0 / 11.0), z_squared);
        let pair_3 = vfmaq_f64(vdupq_n_f64(1.0 / 13.0), vdupq_n_f64(1.0 / 15.0), z_squared);
        let low = vfmaq_f64(pair_0, pair_1, z_fourth);
        let high = vfmaq_f64(pair_2, pair_3, z_fourth);
        let polynomial = vfmaq_f64(low, high, z_eighth);
        let mantissa_log = vmulq_n_f64(vmulq_f64(z, polynomial), 2.0);
        let result = vfmaq_n_f64(mantissa_log, exponent, 0.693_145_751_953_125);
        let multiply_add: unsafe fn(float64x2_t, float64x2_t, f64) -> float64x2_t = vfmaq_n_f64;
        multiply_add(result, exponent, 1.428_606_820_309_417_3e-6)
    }
}

/// Exponential for the finite f64 range generated by Tweedie powers of f32
/// predictions. Range reduction is followed by a degree-12 polynomial on
/// [-ln(2)/2, ln(2)/2].
#[inline]
#[target_feature(enable = "neon")]
unsafe fn expq_f64(value: float64x2_t) -> float64x2_t {
    // SAFETY: the caller guarantees NEON support; all operations use registers.
    unsafe {
        let scaled = vmulq_n_f64(value, std::f64::consts::LOG2_E);
        let exponent = vcvtnq_s64_f64(scaled);
        let exponent_f64 = vcvtq_f64_s64(exponent);
        let mut reduced = vfmsq_n_f64(value, exponent_f64, 0.693_145_751_953_125);
        reduced = vfmsq_n_f64(reduced, exponent_f64, 1.428_606_820_309_417_3e-6);

        // Independent polynomial pairs shorten the FMA dependency chain.
        let squared = vmulq_f64(reduced, reduced);
        let fourth = vmulq_f64(squared, squared);
        let eighth = vmulq_f64(fourth, fourth);
        let pair_0 = vaddq_f64(vdupq_n_f64(1.0), reduced);
        let pair_1 = vfmaq_f64(vdupq_n_f64(0.5), vdupq_n_f64(1.0 / 6.0), reduced);
        let pair_2 = vfmaq_f64(vdupq_n_f64(1.0 / 24.0), vdupq_n_f64(1.0 / 120.0), reduced);
        let pair_3 = vfmaq_f64(
            vdupq_n_f64(1.0 / 720.0),
            vdupq_n_f64(1.0 / 5_040.0),
            reduced,
        );
        let pair_4 = vfmaq_f64(
            vdupq_n_f64(1.0 / 40_320.0),
            vdupq_n_f64(1.0 / 362_880.0),
            reduced,
        );
        let pair_5 = vfmaq_f64(
            vdupq_n_f64(1.0 / 3_628_800.0),
            vdupq_n_f64(1.0 / 39_916_800.0),
            reduced,
        );
        let low = vfmaq_f64(pair_0, pair_1, squared);
        let middle = vfmaq_f64(pair_2, pair_3, squared);
        let high = vfmaq_f64(pair_4, pair_5, squared);
        let low = vfmaq_f64(low, middle, fourth);
        let high = vfmaq_f64(high, vdupq_n_f64(1.0 / 479_001_600.0), fourth);
        let polynomial = vfmaq_f64(low, high, eighth);

        let exponent_bits = vshlq_n_s64::<52>(vaddq_s64(exponent, vdupq_n_s64(1023)));
        let multiply: unsafe fn(float64x2_t, float64x2_t) -> float64x2_t = vmulq_f64;
        multiply(polynomial, vreinterpretq_f64_s64(exponent_bits))
    }
}

/// Number of the 16 `cuts` that are `<= value`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn count_le_16(cuts: &[f32], value: f32) -> usize {
    debug_assert_eq!(cuts.len(), 16);
    // SAFETY: the caller guarantees NEON support and exactly 16 readable
    // values; each load below covers one aligned quarter of them.
    unsafe {
        let value = vdupq_n_f32(value);
        let ptr = cuts.as_ptr();
        let mut count = vdupq_n_u32(0);
        for quarter in 0..4 {
            let mask = vcleq_f32(vld1q_f32(ptr.add(quarter * VECTOR_WIDTH)), value);
            count = vsubq_u32(count, mask);
        }
        vaddvq_u32(count) as usize
    }
}

/// `alpha * h[i] / fma(alpha, u[i], 1)` for eight lanes, four at a time.
#[target_feature(enable = "neon")]
pub(super) unsafe fn edge_terms_8(alpha: f32, h: &[f32; 8], u: &[f32; 8]) -> [f32; 8] {
    let mut out = [0f32; 8];
    // SAFETY: the caller guarantees NEON support; each access covers one
    // half of an eight-lane array.
    unsafe {
        let a = vdupq_n_f32(alpha);
        let one = vdupq_n_f32(1.0);
        for half in 0..2 {
            let offset = half * VECTOR_WIDTH;
            let hv = vld1q_f32(h.as_ptr().add(offset));
            let uv = vld1q_f32(u.as_ptr().add(offset));
            let t = vdivq_f32(vmulq_f32(a, hv), vfmaq_f32(one, a, uv));
            vst1q_f32(out.as_mut_ptr().add(offset), t);
        }
    }
    out
}

/// `c[i] * fma(alpha, u[i], 1)` for eight lanes, four at a time.
#[target_feature(enable = "neon")]
pub(super) unsafe fn scaled_basis_8(alpha: f32, c: &[f32; 8], u: &[f32; 8]) -> [f32; 8] {
    let mut out = [0f32; 8];
    // SAFETY: the caller guarantees NEON support; each access covers one
    // half of an eight-lane array.
    unsafe {
        let a = vdupq_n_f32(alpha);
        let one = vdupq_n_f32(1.0);
        for half in 0..2 {
            let offset = half * VECTOR_WIDTH;
            let cv = vld1q_f32(c.as_ptr().add(offset));
            let uv = vld1q_f32(u.as_ptr().add(offset));
            vst1q_f32(
                out.as_mut_ptr().add(offset),
                vmulq_f32(cv, vfmaq_f32(one, a, uv)),
            );
        }
    }
    out
}

/// `c[i] / fma(alpha, u[i], 1)` for eight lanes when every `c[i]` and
/// every denominator is finite, else `None`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn divided_basis_8(alpha: f32, c: &[f32; 8], u: &[f32; 8]) -> Option<[f32; 8]> {
    let mut out = [0f32; 8];
    // SAFETY: the caller guarantees NEON support; each access covers one
    // half of an eight-lane array.
    unsafe {
        let a = vdupq_n_f32(alpha);
        let one = vdupq_n_f32(1.0);
        let inf = vdupq_n_f32(f32::INFINITY);
        let mut finite = vdupq_n_u32(u32::MAX);
        for half in 0..2 {
            let offset = half * VECTOR_WIDTH;
            let cv = vld1q_f32(c.as_ptr().add(offset));
            let uv = vld1q_f32(u.as_ptr().add(offset));
            let old = vfmaq_f32(one, a, uv);
            // `|x| < inf` is false exactly for infinities and NaN.
            finite = vandq_u32(finite, vcltq_f32(vabsq_f32(cv), inf));
            finite = vandq_u32(finite, vcltq_f32(vabsq_f32(old), inf));
            vst1q_f32(out.as_mut_ptr().add(offset), vdivq_f32(cv, old));
        }
        (vminvq_u32(finite) == u32::MAX).then_some(out)
    }
}

/// Vector-loop shell of a `&mut [f32]` unary inplace kernel: vector fast path
/// for regular lanes, scalar per-lane fallback otherwise. The kernel and
/// scalar formulas (intrinsics included) are passed in as expressions.
macro_rules! unary_inplace_kernel {
    ($name:ident, $kernel:expr, $scalar:expr) => {
        #[target_feature(enable = "neon")]
        pub(super) unsafe fn $name(values: &mut [f32]) {
            // SAFETY: the caller guarantees NEON support. Pointer bounds are
            // documented at each memory access below.
            unsafe {
                let mut index = 0;
                while index + VECTOR_WIDTH <= values.len() {
                    // SAFETY: the loop condition leaves four readable and writable values.
                    let input = vld1q_f32(values.as_ptr().add(index));
                    if regular_input(input) {
                        // SAFETY: the loop condition leaves four writable values.
                        vst1q_f32(values.as_mut_ptr().add(index), ($kernel)(input));
                    } else {
                        for value in &mut values[index..index + VECTOR_WIDTH] {
                            *value = ($scalar)(*value);
                        }
                    }
                    index += VECTOR_WIDTH;
                }
                for value in &mut values[index..] {
                    *value = ($scalar)(*value);
                }
            }
        }
    };
}

unary_inplace_kernel!(exp_inplace, expq_f32::<true>, f32::exp);
unary_inplace_kernel!(sigmoid_inplace, sigmoidq_f32, sigmoid_scalar);

#[target_feature(enable = "neon")]
pub(super) unsafe fn logistic_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let scale = vdupq_n_f32(scale_pos_weight);
        let min_hess_vector = vdupq_n_f32(min_hess);
        let scalar_range = |out: &mut [GradPair], range| {
            scalar::logistic_gradient(
                preds,
                labels,
                weights,
                scale_pos_weight,
                min_hess,
                out,
                range,
            );
        };
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the loop condition and objective length checks leave four
            // readable inputs and four writable GradPair outputs.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            gradient_guard!(
                !regular_input(pred),
                index,
                scalar_range(out, index..index + VECTOR_WIDTH)
            );
            let label = vld1q_f32(labels.as_ptr().add(index));
            let probability = sigmoidq_f32(pred);
            let mut weight = match weights {
                Some(values) => vld1q_f32(values.as_ptr().add(index)),
                None => one,
            };
            weight = vmulq_f32(weight, vbslq_f32(vceqq_f32(label, one), scale, one));
            let grad = vmulq_f32(vsubq_f32(probability, label), weight);
            let hess = vmulq_f32(
                vmaxq_f32(
                    vmulq_f32(probability, vsubq_f32(one, probability)),
                    min_hess_vector,
                ),
                weight,
            );
            // SAFETY: the loop condition leaves room for four pairs; see `store_grad_pairs`.
            store_grad_pairs(out.as_mut_ptr(), index, grad, hess);
            index += VECTOR_WIDTH;
        }

        scalar_range(out, index..preds.len());
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn poisson_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    max_delta_step: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let delta = vdupq_n_f32(max_delta_step);
        let mut index = 0;
        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable inputs.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            let shifted = vaddq_f32(pred, delta);
            gradient_guard!(
                !regular_input(pred) || !regular_input(shifted),
                index,
                scalar::poisson_gradient(
                    preds,
                    labels,
                    weights,
                    max_delta_step,
                    out,
                    index..index + VECTOR_WIDTH,
                )
            );
            let label = vld1q_f32(labels.as_ptr().add(index));
            let weight = match weights {
                Some(values) => vld1q_f32(values.as_ptr().add(index)),
                None => vdupq_n_f32(1.0),
            };
            let grad = vmulq_f32(vsubq_f32(expq_f32::<true>(pred), label), weight);
            let hess = vmulq_f32(expq_f32::<true>(shifted), weight);
            // SAFETY: the loop condition leaves room for four pairs; see `store_grad_pairs`.
            store_grad_pairs(out.as_mut_ptr(), index, grad, hess);
            index += VECTOR_WIDTH;
        }
        scalar::poisson_gradient(
            preds,
            labels,
            weights,
            max_delta_step,
            out,
            index..preds.len(),
        );
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn gamma_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let scale = vdupq_n_f32(scale_pos_weight);
        let scalar_range = |out: &mut [GradPair], range| {
            scalar::gamma_gradient(preds, labels, weights, scale_pos_weight, out, range);
        };
        let mut index = 0;
        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable inputs.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            let negative = vnegq_f32(pred);
            gradient_guard!(
                !regular_input(negative),
                index,
                scalar_range(out, index..index + VECTOR_WIDTH)
            );
            let label = vld1q_f32(labels.as_ptr().add(index));
            let mut weight = match weights {
                Some(values) => vld1q_f32(values.as_ptr().add(index)),
                None => one,
            };
            weight = vmulq_f32(weight, vbslq_f32(vceqq_f32(label, one), scale, one));
            let scaled = vmulq_f32(label, expq_f32::<true>(negative));
            let grad = vmulq_f32(vsubq_f32(one, scaled), weight);
            let hess = vmulq_f32(scaled, weight);
            // SAFETY: the loop condition leaves room for four pairs; see `store_grad_pairs`.
            store_grad_pairs(out.as_mut_ptr(), index, grad, hess);
            index += VECTOR_WIDTH;
        }
        scalar_range(out, index..preds.len());
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn tweedie_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    rho: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let one_minus_rho = vdupq_n_f32(1.0 - rho);
        let two_minus_rho = vdupq_n_f32(2.0 - rho);
        let mut index = 0;
        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable inputs.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            let input_1 = vmulq_f32(one_minus_rho, pred);
            let input_2 = vmulq_f32(two_minus_rho, pred);
            gradient_guard!(
                !regular_input(input_1) || !regular_input(input_2),
                index,
                scalar::tweedie_gradient(
                    preds,
                    labels,
                    weights,
                    rho,
                    out,
                    index..index + VECTOR_WIDTH,
                )
            );
            let label = vld1q_f32(labels.as_ptr().add(index));
            let weight = match weights {
                Some(values) => vld1q_f32(values.as_ptr().add(index)),
                None => vdupq_n_f32(1.0),
            };
            let exp_1 = expq_f32::<true>(input_1);
            let exp_2 = expq_f32::<true>(input_2);
            let label_exp_1 = vmulq_f32(label, exp_1);
            let grad = vmulq_f32(vsubq_f32(exp_2, label_exp_1), weight);
            let hess = vmulq_f32(
                vsubq_f32(
                    vmulq_f32(two_minus_rho, exp_2),
                    vmulq_f32(one_minus_rho, label_exp_1),
                ),
                weight,
            );
            // SAFETY: the loop condition leaves room for four pairs; see `store_grad_pairs`.
            store_grad_pairs(out.as_mut_ptr(), index, grad, hess);
            index += VECTOR_WIDTH;
        }
        scalar::tweedie_gradient(preds, labels, weights, rho, out, index..preds.len());
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_inplace(values: &mut [f32]) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let Some((min, max)) = finite_min_max(values) else {
            super::softmax_scalar(values);
            return;
        };
        if max - min > MAX_FAST_EXP_INPUT {
            super::softmax_scalar(values);
            return;
        }

        let max_vector = vdupq_n_f32(max);
        let mut sum_vector = vdupq_n_f32(0.0);
        let mut index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable and writable values.
            let input = vld1q_f32(values.as_ptr().add(index));
            let exp = expq_f32::<false>(vsubq_f32(input, max_vector));
            sum_vector = vaddq_f32(sum_vector, exp);
            // SAFETY: the loop condition leaves four writable values.
            vst1q_f32(values.as_mut_ptr().add(index), exp);
            index += VECTOR_WIDTH;
        }
        let mut sum = vaddvq_f32(sum_vector);
        for value in &mut values[index..] {
            *value = (*value - max).exp();
            sum += *value;
        }

        let inverse = 1.0 / sum;
        let inverse_vector = vdupq_n_f32(inverse);
        index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable and writable values.

            let probability = vld1q_f32(values.as_ptr().add(index));
            vst1q_f32(
                values.as_mut_ptr().add(index),
                vmulq_f32(probability, inverse_vector),
            );

            index += VECTOR_WIDTH;
        }
        for value in &mut values[index..] {
            *value *= inverse;
        }
    }
}

/// Process four independent rows in the lanes, so short rows need neither a
/// horizontal reduction nor a scratch pass through the output matrix.
///
/// `GRADIENT` selects the shift of `SoftmaxMultiClassObj::GetGradient`,
/// `max(f32::MIN_POSITIVE, row...)`, instead of the plain row maximum of
/// `common::Softmax`; rows whose maximum is at least `MIN_POSITIVE` are
/// unaffected.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn short_softmax_batch<const K: usize, const GRADIENT: bool>(
    preds: &[f32],
) -> Option<[float32x4_t; K]> {
    // SAFETY: callers provide four complete rows, K is 2, 3, or 4, and NEON
    // is available. Interleaved loads transpose row-major inputs into classes.
    unsafe {
        let mut values = [vdupq_n_f32(0.0); K];
        match K {
            2 => {
                let input = vld2q_f32(preds.as_ptr());
                values[0] = input.0;
                values[1] = input.1;
            }
            3 => {
                let input = vld3q_f32(preds.as_ptr());
                values[0] = input.0;
                values[1] = input.1;
                values[2] = input.2;
            }
            4 => {
                let input = vld4q_f32(preds.as_ptr());
                values[0] = input.0;
                values[1] = input.1;
                values[2] = input.2;
                values[3] = input.3;
            }
            _ => unreachable!(),
        }
        let mut minimum = values[0];
        let mut maximum = values[0];
        for &value in &values[1..] {
            minimum = vminq_f32(minimum, value);
            maximum = vmaxq_f32(maximum, value);
        }
        if GRADIENT {
            // `vmaxq_f32` propagates NaN, so the range guard below still
            // rejects non-finite rows.
            maximum = vmaxq_f32(maximum, vdupq_n_f32(f32::MIN_POSITIVE));
        }
        // NaNs propagate through min/max; infinities produce a non-finite
        // range. Either makes this ordered comparison fail for that row.
        if vminvq_u32(vcleq_f32(
            vsubq_f32(maximum, minimum),
            vdupq_n_f32(MAX_FAST_EXP_INPUT),
        )) != u32::MAX
        {
            return None;
        }
        let mut sum = vdupq_n_f32(0.0);
        for value in &mut values {
            *value = expq_f32::<true>(vsubq_f32(*value, maximum));
            sum = vaddq_f32(sum, *value);
        }
        let inverse = vdivq_f32(vdupq_n_f32(1.0), sum);
        for value in &mut values {
            *value = vmulq_f32(*value, inverse);
        }
        Some(values)
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn short_softmax_rows<const K: usize>(values: &mut [f32]) {
    // SAFETY: NEON is available, K is 2, 3, or 4, and chunks bound each
    // interleaved load/store to four complete rows. Remaining rows are scalar.
    unsafe {
        #[allow(
            clippy::chunks_exact_to_as_chunks,
            reason = "`as_chunks_mut::<{ 4 * K }>` needs generic_const_exprs"
        )]
        let mut batches = values.chunks_exact_mut(4 * K);
        for batch in &mut batches {
            if let Some(p) = short_softmax_batch::<K, false>(batch) {
                match K {
                    2 => vst2q_f32(batch.as_mut_ptr(), float32x4x2_t(p[0], p[1])),
                    3 => vst3q_f32(batch.as_mut_ptr(), float32x4x3_t(p[0], p[1], p[2])),
                    4 => vst4q_f32(batch.as_mut_ptr(), float32x4x4_t(p[0], p[1], p[2], p[3])),
                    _ => unreachable!(),
                }
            } else {
                for row in batch.chunks_mut(K) {
                    super::softmax_scalar(row);
                }
            }
        }
        for row in batches.into_remainder().chunks_mut(K) {
            super::softmax_scalar(row);
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn short_softmax_gradient<const K: usize>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: dispatch checks complete matrices and NEON support. Each batch
    // covers four labels/weights and 4*K predictions/output pairs; K is 2..=4.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        let minimum = vdupq_n_f32(min_hess);
        let mut row = 0;
        while row + 4 <= labels.len() {
            let base = row * K;
            if let Some(probabilities) = short_softmax_batch::<K, true>(&preds[base..base + 4 * K])
            {
                // Saturating conversion matches Rust's float-to-usize cast
                // when comparing with class IDs 0..K, including NaN/negatives.
                let target = vcvtq_u32_f32(vld1q_f32(labels.as_ptr().add(row)));
                let weight = match weights {
                    Some(weights) => vld1q_f32(weights.as_ptr().add(row)),
                    None => one,
                };
                let mut low = [vdupq_n_u64(0); K];
                let mut high = low;
                for class in 0..K {
                    let probability = probabilities[class];
                    let indicator =
                        vbslq_f32(vceqq_u32(target, vdupq_n_u32(class as u32)), one, zero);
                    let gradient = vmulq_f32(vsubq_f32(probability, indicator), weight);
                    let hessian = vmaxnmq_f32(
                        vmulq_f32(
                            vmulq_f32(vmulq_n_f32(probability, 2.0), vsubq_f32(one, probability)),
                            weight,
                        ),
                        minimum,
                    );
                    low[class] = vreinterpretq_u64_f32(vzip1q_f32(gradient, hessian));
                    high[class] = vreinterpretq_u64_f32(vzip2q_f32(gradient, hessian));
                }
                // Each u64 lane contains one repr(C) GradPair. Interleaving
                // classes restores row-major order, two complete rows per store.
                for (offset, pairs) in [(0, low), (2 * K, high)] {
                    #[allow(
                        clippy::cast_ptr_alignment,
                        reason = "NEON st2/st4 stores accept any alignment"
                    )]
                    let dest = out.as_mut_ptr().add(base + offset).cast::<u64>();
                    match K {
                        2 => vst2q_u64(dest, uint64x2x2_t(pairs[0], pairs[1])),
                        3 => {
                            // GradPair guarantees only f32 alignment. Avoid
                            // vst3q_u64: stdarch can implement it using a typed
                            // copy that requires an eight-byte-aligned pointer.
                            // Assemble [a0,b0,c0,a1,b1,c1] in three vectors.
                            let dest = dest.cast::<f32>();
                            vst1q_f32(dest, vreinterpretq_f32_u64(vzip1q_u64(pairs[0], pairs[1])));
                            vst1q_f32(
                                dest.add(4),
                                vreinterpretq_f32_u64(vzip1q_u64(
                                    pairs[2],
                                    vextq_u64::<1>(pairs[0], pairs[0]),
                                )),
                            );
                            vst1q_f32(
                                dest.add(8),
                                vreinterpretq_f32_u64(vzip2q_u64(pairs[1], pairs[2])),
                            );
                        }
                        4 => vst4q_u64(dest, uint64x2x4_t(pairs[0], pairs[1], pairs[2], pairs[3])),
                        _ => unreachable!(),
                    }
                }
            } else {
                super::softmax_gradient_rows_scalar(
                    preds,
                    labels,
                    weights,
                    min_hess,
                    out,
                    row..row + 4,
                    K,
                );
            }
            row += 4;
        }
        super::softmax_gradient_rows_scalar(
            preds,
            labels,
            weights,
            min_hess,
            out,
            row..labels.len(),
            K,
        );
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_rows_inplace(values: &mut [f32], num_class: usize) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        for row in values.chunks_mut(num_class) {
            // SAFETY: this function may only be entered after NEON detection and
            // `chunks_mut` bounds every class row.
            softmax_inplace(row);
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn softmax_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        for row in 0..labels.len() {
            let base = row * num_class;
            // SAFETY: the dispatcher's complete-matrix check ensures these row
            // slices exist, and NEON was detected before this function was called.

            softmax_gradient_row(
                &preds[base..base + num_class],
                labels[row] as usize,
                weights.map_or(1.0, |values| values[row]),
                min_hess,
                &mut out[base..base + num_class],
            );
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn softmax_gradient_row(
    preds: &[f32],
    label: usize,
    weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let Some((min, max)) = finite_min_max(preds) else {
            super::softmax_gradient_row_scalar(preds, label, weight, min_hess, out);
            return;
        };
        // `SoftmaxMultiClassObj::GetGradient` seeds `wmax` with
        // `numeric_limits<float>::min()`; a row entirely below it (all large
        // negative margins) then spans more than the fast-exp range and takes
        // the scalar path, which underflows to `0/0` exactly as XGBoost does.
        let max = max.max(f32::MIN_POSITIVE);
        if max - min > MAX_FAST_EXP_INPUT {
            super::softmax_gradient_row_scalar(preds, label, weight, min_hess, out);
            return;
        }

        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let max_vector = vdupq_n_f32(max);
        let mut sum_vector = zero;
        let mut index = 0;
        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the loop condition leaves four inputs and four output pairs.
            let input = vld1q_f32(preds.as_ptr().add(index));
            let exp = expq_f32::<true>(vsubq_f32(input, max_vector));
            sum_vector = vaddq_f32(sum_vector, exp);
            // Store the exponent in the gradient field as scratch space. The second
            // pass overwrites both fields with their final values.

            vst2q_f32(
                out.as_mut_ptr().add(index).cast::<f32>(),
                float32x4x2_t(exp, zero),
            );
            index += VECTOR_WIDTH;
        }
        let vector_end = index;
        let mut sum = vaddvq_f32(sum_vector);
        for index in vector_end..preds.len() {
            let exp = (preds[index] - max).exp();
            out[index].grad = exp;
            sum += exp;
        }

        let inverse = vdupq_n_f32(1.0 / sum);
        let row_weight = vdupq_n_f32(weight);
        let min_hessian = vdupq_n_f32(min_hess);
        let class_offsets_data = [0_u32, 1, 2, 3];
        // SAFETY: the local array contains exactly four u32 lanes.
        let class_offsets = vld1q_u32(class_offsets_data.as_ptr());
        let target_class = vdupq_n_u32(label as u32);
        index = 0;
        while index < vector_end {
            // SAFETY: the first pass initialized four GradPairs at this index.
            let scratch = vld2q_f32(out.as_ptr().add(index).cast::<f32>());
            let probability = vmulq_f32(scratch.0, inverse);
            let classes = vaddq_u32(vdupq_n_u32(index as u32), class_offsets);
            let target = vbslq_f32(vceqq_u32(classes, target_class), one, zero);
            let gradient = vmulq_f32(vsubq_f32(probability, target), row_weight);
            let hessian = vmaxq_f32(
                vmulq_f32(
                    vmulq_n_f32(vmulq_f32(probability, vsubq_f32(one, probability)), 2.0),
                    row_weight,
                ),
                min_hessian,
            );
            // SAFETY: the loop bounds leave four writable output pairs.

            vst2q_f32(
                out.as_mut_ptr().add(index).cast::<f32>(),
                float32x4x2_t(gradient, hessian),
            );
            index += VECTOR_WIDTH;
        }
        let inverse = 1.0 / sum;
        for (class, output) in out.iter_mut().enumerate().skip(vector_end) {
            let probability = output.grad * inverse;
            let target = if class == label { 1.0 } else { 0.0 };
            *output = GradPair::new(
                (probability - target) * weight,
                (2.0 * probability * (1.0 - probability) * weight).max(min_hess),
            );
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn distance_sum<const SQUARED: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let mut sum_low = vdupq_n_f64(0.0);
        let mut sum_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable values.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            let label = vld1q_f32(labels.as_ptr().add(index));
            let difference_low = vsubq_f64(
                vcvt_f64_f32(vget_low_f32(pred)),
                vcvt_f64_f32(vget_low_f32(label)),
            );
            let difference_high = vsubq_f64(vcvt_high_f64_f32(pred), vcvt_high_f64_f32(label));
            let distance_low = if SQUARED {
                vmulq_f64(difference_low, difference_low)
            } else {
                vabsq_f64(difference_low)
            };
            let distance_high = if SQUARED {
                vmulq_f64(difference_high, difference_high)
            } else {
                vabsq_f64(difference_high)
            };
            accumulate_metric_sum!(
                weights,
                index,
                distance_low,
                distance_high,
                sum_low,
                sum_high,
                weight_low,
                weight_high
            );
            index += VECTOR_WIDTH;
        }

        let tail = scalar::distance_sum::<SQUARED>(preds, labels, weights, index..preds.len());
        let sum = vaddvq_f64(vaddq_f64(sum_low, sum_high)) + tail.0;
        let weight_sum = match weights {
            Some(_) => vaddvq_f64(vaddq_f64(weight_low, weight_high)) + tail.1,
            None => index as f64 + tail.1,
        };
        (sum, weight_sum)
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn classification_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let threshold = vdupq_n_f32(0.5);
        let mut index = 0;
        if let Some(weights) = weights {
            let mut wrong_low = vdupq_n_f64(0.0);
            let mut wrong_high = vdupq_n_f64(0.0);
            let mut weight_low = vdupq_n_f64(0.0);
            let mut weight_high = vdupq_n_f64(0.0);
            while index + VECTOR_WIDTH <= preds.len() {
                // SAFETY: the common-length contract leaves four values in
                // each input slice.
                let pred = vld1q_f32(preds.as_ptr().add(index));
                let label = vld1q_f32(labels.as_ptr().add(index));
                let weight = lane_weights(weights, index);
                let mismatch = veorq_u32(vcgtq_f32(pred, threshold), vcgtq_f32(label, threshold));
                let wrong_weight = vbslq_f32(mismatch, weight, vdupq_n_f32(0.0));
                let current_wrong_low = vcvt_f64_f32(vget_low_f32(wrong_weight));
                let current_wrong_high = vcvt_high_f64_f32(wrong_weight);
                let current_weight_low = vcvt_f64_f32(vget_low_f32(weight));
                let current_weight_high = vcvt_high_f64_f32(weight);
                wrong_low = vaddq_f64(wrong_low, current_wrong_low);
                wrong_high = vaddq_f64(wrong_high, current_wrong_high);
                weight_low = vaddq_f64(weight_low, current_weight_low);
                weight_high = vaddq_f64(weight_high, current_weight_high);
                index += VECTOR_WIDTH;
            }
            let tail =
                scalar::classification_error_sum(preds, labels, Some(weights), index..preds.len());
            let wrong = vaddvq_f64(vaddq_f64(wrong_low, wrong_high)) + tail.0;
            let weight_sum = vaddvq_f64(vaddq_f64(weight_low, weight_high)) + tail.1;
            (wrong, weight_sum)
        } else {
            let one = vdupq_n_u32(1);
            let mut wrong_low = vdupq_n_u64(0);
            let mut wrong_high = vdupq_n_u64(0);
            while index + VECTOR_WIDTH <= preds.len() {
                // SAFETY: the common-length contract leaves four values in
                // each input slice.
                let pred = vld1q_f32(preds.as_ptr().add(index));
                let label = vld1q_f32(labels.as_ptr().add(index));
                let mismatch = vandq_u32(
                    veorq_u32(vcgtq_f32(pred, threshold), vcgtq_f32(label, threshold)),
                    one,
                );
                wrong_low = vaddq_u64(wrong_low, vmovl_u32(vget_low_u32(mismatch)));
                wrong_high = vaddq_u64(wrong_high, vmovl_high_u32(mismatch));
                index += VECTOR_WIDTH;
            }
            let tail = scalar::classification_error_sum(preds, labels, None, index..preds.len());
            let wrong = vaddvq_u64(vaddq_u64(wrong_low, wrong_high)) as f64 + tail.0;
            (wrong, preds.len() as f64)
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn log_loss_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let one = vdupq_n_f64(1.0);
        let floor = vdupq_n_f64(BINARY_LOG_LOSS_EPSILON);
        let mut loss_low = vdupq_n_f64(0.0);
        let mut loss_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut fallback = (0.0, 0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable values.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            metric_finite_guard!(
                pred,
                index,
                fallback,
                scalar::log_loss(preds, labels, weights, index..index + VECTOR_WIDTH)
            );
            let label = vld1q_f32(labels.as_ptr().add(index));
            // XGBoost floors each log argument at ε separately and does not
            // clamp the prediction into [0, 1]; the finite guard above makes
            // both logs finite, so a zero label coefficient contributes 0
            // exactly as upstream's `xlogy` branch does.
            let probability_low = vcvt_f64_f32(vget_low_f32(pred));
            let probability_high = vcvt_high_f64_f32(pred);
            let label_low = vcvt_f64_f32(vget_low_f32(label));
            let label_high = vcvt_high_f64_f32(label);
            let value_low = vnegq_f64(vaddq_f64(
                vmulq_f64(label_low, logq_f64(vmaxq_f64(probability_low, floor))),
                vmulq_f64(
                    vsubq_f64(one, label_low),
                    logq_f64(vmaxq_f64(vsubq_f64(one, probability_low), floor)),
                ),
            ));
            let value_high = vnegq_f64(vaddq_f64(
                vmulq_f64(label_high, logq_f64(vmaxq_f64(probability_high, floor))),
                vmulq_f64(
                    vsubq_f64(one, label_high),
                    logq_f64(vmaxq_f64(vsubq_f64(one, probability_high), floor)),
                ),
            ));
            accumulate_metric_sum!(
                weights,
                index,
                value_low,
                value_high,
                loss_low,
                loss_high,
                weight_low,
                weight_high
            );
            index += VECTOR_WIDTH;
        }

        let tail = scalar::log_loss(preds, labels, weights, index..preds.len());
        finish_metric_sum!(
            loss_low,
            loss_high,
            weight_low,
            weight_high,
            fallback,
            tail,
            weights,
            preds.len()
        )
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn positive_nloglik_sum<const GAMMA: bool>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let minimum = vdupq_n_f64(MIN_POSITIVE_PREDICTION);
        let mut loss_low = vdupq_n_f64(0.0);
        let mut loss_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut fallback = (0.0, 0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable values.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            metric_finite_guard!(
                pred,
                index,
                fallback,
                scalar::positive_nloglik::<GAMMA>(
                    preds,
                    labels,
                    weights,
                    index..index + VECTOR_WIDTH
                )
            );
            let label = vld1q_f32(labels.as_ptr().add(index));
            let prediction_low = vmaxq_f64(vcvt_f64_f32(vget_low_f32(pred)), minimum);
            let prediction_high = vmaxq_f64(vcvt_high_f64_f32(pred), minimum);
            let label_low = vcvt_f64_f32(vget_low_f32(label));
            let label_high = vcvt_high_f64_f32(label);
            let log_low = logq_f64(prediction_low);
            let log_high = logq_f64(prediction_high);
            let value_low = if GAMMA {
                vaddq_f64(vdivq_f64(label_low, prediction_low), log_low)
            } else {
                vsubq_f64(prediction_low, vmulq_f64(label_low, log_low))
            };
            let value_high = if GAMMA {
                vaddq_f64(vdivq_f64(label_high, prediction_high), log_high)
            } else {
                vsubq_f64(prediction_high, vmulq_f64(label_high, log_high))
            };
            accumulate_metric_sum!(
                weights,
                index,
                value_low,
                value_high,
                loss_low,
                loss_high,
                weight_low,
                weight_high
            );
            index += VECTOR_WIDTH;
        }

        let tail = scalar::positive_nloglik::<GAMMA>(preds, labels, weights, index..preds.len());
        finish_metric_sum!(
            loss_low,
            loss_high,
            weight_low,
            weight_high,
            fallback,
            tail,
            weights,
            preds.len()
        )
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn tweedie_nloglik_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<RowWeights<'_>>,
    rho: f64,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let minimum = vdupq_n_f64(MIN_POSITIVE_PREDICTION);
        let first_power = vdupq_n_f64(1.0 - rho);
        let second_power = vdupq_n_f64(2.0 - rho);
        let mut loss_low = vdupq_n_f64(0.0);
        let mut loss_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut fallback = (0.0, 0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= preds.len() {
            // SAFETY: the common-length contract leaves four readable values.
            let pred = vld1q_f32(preds.as_ptr().add(index));
            metric_finite_guard!(
                pred,
                index,
                fallback,
                scalar::tweedie_nloglik(preds, labels, weights, rho, index..index + VECTOR_WIDTH)
            );
            let label = vld1q_f32(labels.as_ptr().add(index));
            let prediction_low = vmaxq_f64(vcvt_f64_f32(vget_low_f32(pred)), minimum);
            let prediction_high = vmaxq_f64(vcvt_high_f64_f32(pred), minimum);
            let label_low = vcvt_f64_f32(vget_low_f32(label));
            let label_high = vcvt_high_f64_f32(label);
            let log_low = logq_f64(prediction_low);
            let log_high = logq_f64(prediction_high);
            // `first_power` and `second_power` are also the denominators.
            let first_low = vdivq_f64(
                vmulq_f64(label_low, expq_f64(vmulq_f64(first_power, log_low))),
                first_power,
            );
            let first_high = vdivq_f64(
                vmulq_f64(label_high, expq_f64(vmulq_f64(first_power, log_high))),
                first_power,
            );
            let second_low = vdivq_f64(expq_f64(vmulq_f64(second_power, log_low)), second_power);
            let second_high = vdivq_f64(expq_f64(vmulq_f64(second_power, log_high)), second_power);
            let value_low = vsubq_f64(second_low, first_low);
            let value_high = vsubq_f64(second_high, first_high);
            accumulate_metric_sum!(
                weights,
                index,
                value_low,
                value_high,
                loss_low,
                loss_high,
                weight_low,
                weight_high
            );
            index += VECTOR_WIDTH;
        }

        let tail = scalar::tweedie_nloglik(preds, labels, weights, rho, index..preds.len());
        finish_metric_sum!(
            loss_low,
            loss_high,
            weight_low,
            weight_high,
            fallback,
            tail,
            weights,
            preds.len()
        )
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn multiclass_log_loss_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let lower = vdupq_n_f64(LOG_LOSS_EPSILON);
        let upper = vdupq_n_f64(1.0);
        let mut loss_low = vdupq_n_f64(0.0);
        let mut loss_high = vdupq_n_f64(0.0);
        let mut weight_low = vdupq_n_f64(0.0);
        let mut weight_high = vdupq_n_f64(0.0);
        let mut fallback = (0.0, 0.0);
        let mut index = 0;

        while index + VECTOR_WIDTH <= labels.len() {
            let selected = [
                preds[index * num_class + labels[index] as usize],
                preds[(index + 1) * num_class + labels[index + 1] as usize],
                preds[(index + 2) * num_class + labels[index + 2] as usize],
                preds[(index + 3) * num_class + labels[index + 3] as usize],
            ];
            // SAFETY: `selected` contains exactly four f32 lanes.
            let probability = vld1q_f32(selected.as_ptr());
            metric_finite_guard!(
                probability,
                index,
                fallback,
                scalar::multiclass_log_loss(
                    preds,
                    labels,
                    weights,
                    num_class,
                    index..index + VECTOR_WIDTH
                )
            );

            let probability_low = vminq_f64(
                vmaxq_f64(vcvt_f64_f32(vget_low_f32(probability)), lower),
                upper,
            );
            let probability_high =
                vminq_f64(vmaxq_f64(vcvt_high_f64_f32(probability), lower), upper);
            let value_low = vnegq_f64(logq_f64(probability_low));
            let value_high = vnegq_f64(logq_f64(probability_high));
            // SAFETY: the dispatcher verified one weight per label.
            accumulate_metric_sum!(
                weights,
                index,
                value_low,
                value_high,
                loss_low,
                loss_high,
                weight_low,
                weight_high
            );
            index += VECTOR_WIDTH;
        }

        let tail =
            scalar::multiclass_log_loss(preds, labels, weights, num_class, index..labels.len());
        finish_metric_sum!(
            loss_low,
            loss_high,
            weight_low,
            weight_high,
            fallback,
            tail,
            weights,
            labels.len()
        )
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn multiclass_error_sum(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    num_class: usize,
) -> (f64, f64) {
    super::multiclass_error_sum_rows(preds, labels, weights, num_class, |row| {
        // SAFETY: the dispatcher verified NEON support, a complete prediction
        // matrix, and a u32-fitting class count.
        unsafe { argmax(row) }
    })
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn argmax(values: &[f32]) -> usize {
    // SAFETY: the caller guarantees NEON support. Pointer bounds are
    // documented at each memory access below.
    unsafe {
        let offsets_data = [0_u32, 1, 2, 3];
        // SAFETY: the local array contains four u32 lanes.
        let offsets = vld1q_u32(offsets_data.as_ptr());
        let mut maxima = vdupq_n_f32(f32::NEG_INFINITY);
        let mut best_indices = vdupq_n_u32(0);
        let mut index = 0;
        while index + VECTOR_WIDTH <= values.len() {
            // SAFETY: the loop condition leaves four readable values.
            let value = vld1q_f32(values.as_ptr().add(index));
            if !finite_input(value) {
                return super::argmax_scalar(values);
            }
            let better = vcgtq_f32(value, maxima);
            maxima = vbslq_f32(better, value, maxima);
            let indices = vaddq_u32(vdupq_n_u32(index as u32), offsets);
            best_indices = vbslq_u32(better, indices, best_indices);
            index += VECTOR_WIDTH;
        }

        let mut lane_maxima = [0.0_f32; VECTOR_WIDTH];
        let mut lane_indices = [0_u32; VECTOR_WIDTH];
        // SAFETY: each output array contains four writable lanes.

        vst1q_f32(lane_maxima.as_mut_ptr(), maxima);
        vst1q_u32(lane_indices.as_mut_ptr(), best_indices);

        let mut max = lane_maxima[0];
        let mut best = lane_indices[0] as usize;
        for lane in 1..VECTOR_WIDTH {
            if lane_maxima[lane] > max
                || (lane_maxima[lane] == max && (lane_indices[lane] as usize) < best)
            {
                max = lane_maxima[lane];
                best = lane_indices[lane] as usize;
            }
        }
        for (tail_index, &value) in values[index..].iter().enumerate() {
            if !value.is_finite() {
                return super::argmax_scalar(values);
            }
            let class = index + tail_index;
            if value > max {
                max = value;
                best = class;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests;
