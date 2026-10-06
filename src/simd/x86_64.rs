//! AVX2/FMA kernels for x86-64. The transcendental kernels mirror the NEON
//! formulas and stay within a few f32 ULPs of the scalar library functions.

use super::{MAX_FAST_EXP_INPUT, scalar, sigmoid_scalar};
use crate::objective::GradPair;
#[allow(
    clippy::wildcard_imports,
    reason = "intrinsic modules are used wholesale"
)]
use std::arch::x86_64::*;

/// f32 lanes per vector.
pub(super) const WIDTH: usize = 8;
// `objective::GRADIENT_BLOCK_ROWS`'s contract: its blocks start on vector blocks.
const _: () = assert!(crate::objective::GRADIENT_BLOCK_ROWS.is_multiple_of(WIDTH));

/// Exponential for finite f32 lanes in [-80, 80]: range reduction to
/// [-ln(2)/2, ln(2)/2] and a seventh-order polynomial (Estrin pairs for
/// latency), as the NEON kernel.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_f32(value: __m256) -> __m256 {
    // SAFETY: the caller guarantees AVX2/FMA support; all operations use
    // registers.
    unsafe {
        let multiply: unsafe fn(__m256, __m256) -> __m256 = _mm256_mul_ps;
        let exponent =
            _mm256_cvtps_epi32(multiply(value, _mm256_set1_ps(std::f32::consts::LOG2_E)));
        let exponent_f32 = _mm256_cvtepi32_ps(exponent);
        // Split ln(2) so the range reduction loses fewer low bits.
        let reduced = _mm256_fnmadd_ps(exponent_f32, _mm256_set1_ps(0.693_359_4), value);
        let reduced = _mm256_fmadd_ps(exponent_f32, _mm256_set1_ps(2.121_944_4e-4), reduced);

        let c = |x: f32| _mm256_set1_ps(x);
        let polynomial = {
            let squared = multiply(reduced, reduced);
            let fourth = multiply(squared, squared);
            let pair_0 = _mm256_add_ps(c(1.0), reduced);
            let pair_1 = _mm256_fmadd_ps(c(1.0 / 6.0), reduced, c(0.5));
            let pair_2 = _mm256_fmadd_ps(c(1.0 / 120.0), reduced, c(1.0 / 24.0));
            let pair_3 = _mm256_fmadd_ps(c(1.0 / 5_040.0), reduced, c(1.0 / 720.0));
            let low = _mm256_fmadd_ps(pair_1, squared, pair_0);
            let high = _mm256_fmadd_ps(pair_3, squared, pair_2);
            _mm256_fmadd_ps(high, fourth, low)
        };

        let exponent_bits =
            _mm256_slli_epi32::<23>(_mm256_add_epi32(exponent, _mm256_set1_epi32(127)));
        multiply(polynomial, _mm256_castsi256_ps(exponent_bits))
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn abs_f32(value: __m256) -> __m256 {
    // SAFETY: the caller guarantees AVX2 support; register-only.
    unsafe {
        let and_not: unsafe fn(__m256, __m256) -> __m256 = _mm256_andnot_ps;
        and_not(_mm256_set1_ps(-0.0), value)
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_f32(value: __m256) -> __m256 {
    // SAFETY: the caller guarantees AVX2/FMA support; register-only.
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let exp = exp_f32(_mm256_sub_ps(_mm256_setzero_ps(), abs_f32(value)));
        let denominator = _mm256_add_ps(one, exp);
        let positive = _mm256_div_ps(one, denominator);
        let negative = _mm256_div_ps(exp, denominator);
        let non_negative = _mm256_cmp_ps::<_CMP_GE_OQ>(value, _mm256_setzero_ps());
        _mm256_blendv_ps(negative, positive, non_negative)
    }
}

/// Whether every lane is finite with magnitude at most [`MAX_FAST_EXP_INPUT`].
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn regular_input(value: __m256) -> bool {
    // SAFETY: the caller guarantees AVX2 support; register-only.
    unsafe {
        let in_range =
            _mm256_cmp_ps::<_CMP_LE_OQ>(abs_f32(value), _mm256_set1_ps(MAX_FAST_EXP_INPUT));
        _mm256_movemask_ps(in_range) == 0xFF
    }
}

/// Store eight `(grad, hess)` pairs row-major as `GradPair`s.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn store_pairs(dest: *mut GradPair, grad: __m256, hess: __m256) {
    // SAFETY: the caller guarantees AVX2 support and eight writable pairs
    // at `dest`; GradPair is repr(C) with two adjacent f32 fields.
    unsafe {
        let low = _mm256_unpacklo_ps(grad, hess); // g0 h0 g1 h1 | g4 h4 g5 h5
        let high = _mm256_unpackhi_ps(grad, hess); // g2 h2 g3 h3 | g6 h6 g7 h7
        let dest = dest.cast::<f32>();
        _mm256_storeu_ps(dest, _mm256_permute2f128_ps::<0x20>(low, high));
        _mm256_storeu_ps(dest.add(WIDTH), _mm256_permute2f128_ps::<0x31>(low, high));
    }
}

/// Vector-loop shell of a `&mut [f32]` unary inplace kernel: vector fast path
/// for regular lanes, scalar per-lane fallback otherwise. The kernel and
/// scalar formulas (intrinsics included) are passed in as expressions.
macro_rules! unary_inplace_kernel {
    ($name:ident, $kernel:expr, $scalar:expr) => {
        #[target_feature(enable = "avx2,fma")]
        pub(super) unsafe fn $name(values: &mut [f32]) {
            // SAFETY: the caller guarantees AVX2/FMA support; every vector
            // access is bounded by the loop condition.
            unsafe {
                let mut index = 0;
                while index + WIDTH <= values.len() {
                    let input = _mm256_loadu_ps(values.as_ptr().add(index));
                    if regular_input(input) {
                        _mm256_storeu_ps(values.as_mut_ptr().add(index), ($kernel)(input));
                    } else {
                        for value in &mut values[index..index + WIDTH] {
                            *value = ($scalar)(*value);
                        }
                    }
                    index += WIDTH;
                }
                for value in &mut values[index..] {
                    *value = ($scalar)(*value);
                }
            }
        }
    };
}

unary_inplace_kernel!(exp_inplace, exp_f32, f32::exp);
unary_inplace_kernel!(sigmoid_inplace, sigmoid_f32, sigmoid_scalar);

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn logistic_gradient(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    scale_pos_weight: f32,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the caller guarantees AVX2/FMA support and that `labels`,
    // `weights` and `out` cover `preds.len()` elements; the loop condition
    // bounds every vector access.
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let scale = _mm256_set1_ps(scale_pos_weight);
        let min_hess_vector = _mm256_set1_ps(min_hess);
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
        while index + WIDTH <= preds.len() {
            let pred = _mm256_loadu_ps(preds.as_ptr().add(index));
            if !regular_input(pred) {
                scalar_range(out, index..index + WIDTH);
                index += WIDTH;
                continue;
            }
            let label = _mm256_loadu_ps(labels.as_ptr().add(index));
            let probability = sigmoid_f32(pred);
            let mut weight = match weights {
                Some(values) => _mm256_loadu_ps(values.as_ptr().add(index)),
                None => one,
            };
            let positive = _mm256_cmp_ps::<_CMP_EQ_OQ>(label, one);
            weight = _mm256_mul_ps(weight, _mm256_blendv_ps(one, scale, positive));
            let grad = _mm256_mul_ps(_mm256_sub_ps(probability, label), weight);
            let hess = _mm256_mul_ps(
                _mm256_max_ps(
                    _mm256_mul_ps(probability, _mm256_sub_ps(one, probability)),
                    min_hess_vector,
                ),
                weight,
            );
            store_pairs(out.as_mut_ptr().add(index), grad, hess);
            index += WIDTH;
        }
        scalar_range(out, index..preds.len());
    }
}

/// Row-local horizontal reduction of a vector holding `WIDTH / K` complete
/// rows of `K` classes (`K` is 2 or 4): every lane receives its row's result.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_reduce<const K: usize>(
    value: __m256,
    op: unsafe fn(__m256, __m256) -> __m256,
) -> __m256 {
    // SAFETY: the caller guarantees AVX2 support; register-only permutes.
    unsafe {
        // Swap adjacent lanes: pairs are complete rows for K = 2.
        let reduced = op(value, _mm256_permute_ps::<0b10_11_00_01>(value));
        if K == 2 {
            reduced
        } else {
            // Swap the halves of each 128-bit lane: quads are rows for K = 4.
            op(reduced, _mm256_permute_ps::<0b01_00_11_10>(reduced))
        }
    }
}

/// Softmax of the `WIDTH / K` rows held in one vector, or `None` when a row is
/// non-finite or spans more than [`MAX_FAST_EXP_INPUT`].
///
/// `GRADIENT` selects the shift of `SoftmaxMultiClassObj::GetGradient`,
/// `max(f32::MIN_POSITIVE, row...)`, instead of the plain row maximum of
/// `common::Softmax`; rows whose maximum is at least `MIN_POSITIVE` are
/// unaffected.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn short_softmax_batch<const K: usize, const GRADIENT: bool>(
    preds: *const f32,
) -> Option<__m256> {
    // SAFETY: the caller guarantees AVX2/FMA support and `WIDTH` readable
    // values at `preds`.
    unsafe {
        let values = _mm256_loadu_ps(preds);
        let mut maximum = row_reduce::<K>(values, _mm256_max_ps);
        let minimum = row_reduce::<K>(values, _mm256_min_ps);
        if GRADIENT {
            // `maxps` returns its second operand when either is NaN, so keep
            // `maximum` second to preserve NaN lanes for the range guard.
            maximum = _mm256_max_ps(_mm256_set1_ps(f32::MIN_POSITIVE), maximum);
        }
        // NaNs propagate through min/max and infinities give a non-finite
        // range, so this ordered comparison fails for such rows.
        let regular = _mm256_cmp_ps::<_CMP_LE_OQ>(
            _mm256_sub_ps(maximum, minimum),
            _mm256_set1_ps(MAX_FAST_EXP_INPUT),
        );
        if _mm256_movemask_ps(regular) != 0xFF {
            return None;
        }
        let exp = exp_f32(_mm256_sub_ps(values, maximum));
        let sum = row_reduce::<K>(exp, _mm256_add_ps);
        Some(_mm256_mul_ps(exp, _mm256_div_ps(_mm256_set1_ps(1.0), sum)))
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn short_softmax_rows<const K: usize>(values: &mut [f32]) {
    // SAFETY: the caller guarantees AVX2/FMA support and K in {2, 4};
    // `as_chunks_mut` bounds every vector access to complete rows.
    unsafe {
        let (batches, remainder) = values.as_chunks_mut::<WIDTH>();
        for batch in batches {
            match short_softmax_batch::<K, false>(batch.as_ptr()) {
                Some(probabilities) => _mm256_storeu_ps(batch.as_mut_ptr(), probabilities),
                None => {
                    for row in batch.chunks_mut(K) {
                        super::softmax_scalar(row);
                    }
                }
            }
        }
        for row in remainder.chunks_mut(K) {
            super::softmax_scalar(row);
        }
    }
}

/// Broadcast the per-row `values` (labels or weights) of the `WIDTH / K` rows
/// in a batch across their `K` lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "_mm_load_sd has no alignment requirement"
)]
unsafe fn broadcast_rows<const K: usize>(values: *const f32) -> __m256 {
    // SAFETY: the caller guarantees AVX2 support and `WIDTH / K` readable
    // values.
    unsafe {
        let (loaded, index) = if K == 2 {
            (
                _mm_loadu_ps(values),
                _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3),
            )
        } else {
            (
                _mm_castpd_ps(_mm_load_sd(values.cast::<f64>())),
                _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1),
            )
        };
        _mm256_permutevar8x32_ps(_mm256_castps128_ps256(loaded), index)
    }
}

/// `1.0` in the lanes whose class equals `label as usize`, following Rust's
/// saturating float-to-integer cast: `NaN` and negatives select class 0.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn class_indicator<const K: usize>(label: __m256) -> __m256 {
    // SAFETY: the caller guarantees AVX2 support; register-only.
    unsafe {
        let class = if K == 2 {
            _mm256_setr_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        } else {
            _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0)
        };
        let one = _mm256_set1_ps(1.0);
        let next = _mm256_add_ps(class, one);
        // class >= 1: class <= label < class + 1 (false for NaN).
        let in_range = _mm256_and_ps(
            _mm256_cmp_ps::<_CMP_GE_OQ>(label, class),
            _mm256_cmp_ps::<_CMP_LT_OQ>(label, next),
        );
        // class 0: everything not >= 1, including NaN and negatives.
        let is_zero = _mm256_cmp_ps::<_CMP_NGE_UQ>(label, one);
        let class_is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(class, _mm256_setzero_ps());
        let and: unsafe fn(__m256, __m256) -> __m256 = _mm256_and_ps;
        and(_mm256_blendv_ps(in_range, is_zero, class_is_zero), one)
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn short_softmax_gradient<const K: usize>(
    preds: &[f32],
    labels: &[f32],
    weights: Option<&[f32]>,
    min_hess: f32,
    out: &mut [GradPair],
) {
    // SAFETY: the dispatcher checked complete matrices (`labels.len() * K ==
    // preds.len() <= out.len()`, weights cover the labels) and AVX2/FMA
    // support; K is 2 or 4 so a batch is `WIDTH / K` whole rows.
    unsafe {
        const { assert!(K == 2 || K == 4) };
        let rows_per_batch = WIDTH / K;
        let one = _mm256_set1_ps(1.0);
        let minimum = _mm256_set1_ps(min_hess);
        let mut row = 0;
        while row + rows_per_batch <= labels.len() {
            let base = row * K;
            match short_softmax_batch::<K, true>(preds.as_ptr().add(base)) {
                Some(probability) => {
                    let label = broadcast_rows::<K>(labels.as_ptr().add(row));
                    let weight = match weights {
                        Some(values) => broadcast_rows::<K>(values.as_ptr().add(row)),
                        None => one,
                    };
                    let indicator = class_indicator::<K>(label);
                    let grad = _mm256_mul_ps(_mm256_sub_ps(probability, indicator), weight);
                    let hess = _mm256_max_ps(
                        _mm256_mul_ps(
                            _mm256_mul_ps(
                                _mm256_mul_ps(probability, _mm256_set1_ps(2.0)),
                                _mm256_sub_ps(one, probability),
                            ),
                            weight,
                        ),
                        minimum,
                    );
                    store_pairs(out.as_mut_ptr().add(base), grad, hess);
                }
                None => {
                    super::softmax_gradient_rows_scalar(
                        preds,
                        labels,
                        weights,
                        min_hess,
                        out,
                        row..row + rows_per_batch,
                        K,
                    );
                }
            }
            row += rows_per_batch;
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

/// Entries of exactly 16 `cuts` that are `<= value` (ordered comparison).
///
/// # Safety
/// `cuts` must hold exactly 16 values.
pub(super) unsafe fn count_le_16(cuts: &[f32], value: f32) -> usize {
    debug_assert_eq!(cuts.len(), 16);
    // SAFETY: the caller guarantees exactly 16 readable values; each load
    // covers one quarter of them.
    unsafe {
        let value = _mm_set1_ps(value);
        let ptr = cuts.as_ptr();
        let mut count = 0usize;
        for quarter in 0..4 {
            let mask = _mm_cmple_ps(_mm_loadu_ps(ptr.add(quarter * 4)), value);
            count += _mm_movemask_ps(mask).count_ones() as usize;
        }
        count
    }
}
