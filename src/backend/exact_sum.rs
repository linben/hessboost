//! The exactness domain of the GPU backends' integer histogram sums.
//!
//! The CPU accumulates each histogram bin in `f64`: a chain of additions in
//! row order within each fixed block of rows, then the block partials in
//! block order (`tree::hist::CpuBackend`). The GPU instead sums integers:
//! at staging time every value
//! `x` becomes the integer `k = x / u`, where `u = 2^grain` is the largest
//! power of two dividing every value of the slice, the kernels add the
//! `k`s in 64-bit integers (any grouping, any order), and the host scales
//! the bin total back by `u`. [`SumDomain::sums_exact`] decides when the two
//! agree bit for bit, from two statistics of the slice: its largest
//! magnitude and its grain.
//!
//! # Proof
//!
//! Let every value be `k u` with an integer `k`, `|k| <= M = max / u`, and
//! let a node sum `n` values with `n M <= 2^53`. Every partial sum either
//! path forms sums a subset of the values, so it is `K u` with an integer
//! `|K| <= n M <= 2^53`.
//!
//! - Staging: `x * 2^-grain` in `f64` scales by a power of two inside the
//!   `f64` range (`grain` is in `[-149, 127]`), so it is exact, and it is an
//!   integer of magnitude at most `2^53`, so the conversion to `i64` is
//!   exact too.
//! - GPU: the integer partials never exceed `2^53 < 2^63`, so every add is
//!   exact, whatever the grouping; the bin total is `K`, which converts to
//!   `f64` exactly (`|K| <= 2^53`), and multiplying by `u` is exact (a
//!   nonzero result is at least `2^-149`, a normal `f64`).
//! - CPU: every partial sum `K u` with `|K| <= 2^53` is an `f64`, so each
//!   add is exact (within a block's chain, and adding a block's partial to
//!   the running total alike) and the bin holds the same `K u`.
//!
//! The kernels therefore never touch a float, so neither rounding modes nor
//! subnormal flushing matter. The check computes `n M` exactly in integers,
//! so it needs no margin. It is also the widest bound these statistics
//! allow: past `2^53` grains the CPU's sums themselves can round (a test
//! below shows one), and no other grouping reproduces that rounding.

/// The largest exact bin sum, in grains: `f64`'s 53-bit significand.
const MAX_UNITS: u64 = 1 << 53;

/// The magnitude statistics of one gradient component (all gradients, or
/// all Hessians, of a slice) that decide whether its histogram sums are
/// exact on both the CPU and the GPU, and the fixed-point scale the GPU
/// sums them in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SumDomain {
    /// Largest magnitude (0 when every value is zero).
    max: f32,
    /// Exponent `k` of the largest `2^k` every nonzero value is a multiple
    /// of (`i32::MAX` when every value is zero).
    grain: i32,
    /// Whether every value is finite (no NaN, no infinity).
    finite: bool,
}

impl SumDomain {
    /// The statistics of no values: every sum is (vacuously) exact.
    pub(crate) const EMPTY: Self = Self {
        max: 0.0,
        grain: i32::MAX,
        finite: true,
    };

    /// The statistics a device folded (the CUDA `grad_domain` kernel): the
    /// largest magnitude's bits, the smallest grain plus 150 (`u32::MAX`
    /// when every value is zero), and whether every value is finite. The
    /// fold is order-free, so this equals [`of`](Self::of) of the values.
    #[cfg_attr(
        not(all(target_os = "linux", feature = "cuda")),
        allow(dead_code, reason = "only the CUDA backend folds on the device")
    )]
    pub(crate) fn from_device(max_bits: u32, grain_code: u32, finite: bool) -> Self {
        Self {
            max: f32::from_bits(max_bits),
            grain: if grain_code == u32::MAX {
                i32::MAX
            } else {
                grain_code as i32 - 150
            },
            finite,
        }
    }

    /// The statistics of `values`, folded over the row chunks in parallel.
    ///
    /// The fold's result does not depend on the order — `max` of
    /// non-negative magnitudes, `min` of grains, and the finiteness flag are
    /// commutative and associative — so this agrees with [`of`](Self::of)
    /// bit for bit. Staging runs it once per tree on the whole gradient
    /// slice, where one thread's fold is a measurable share of the round.
    pub(crate) fn of_slice<T: Sync>(values: &[T], project: impl Fn(&T) -> f32 + Sync) -> Self {
        use rayon::prelude::*;
        /// Values per fold chunk: enough to amortize the reduction.
        const CHUNK: usize = 4096;
        values
            .par_chunks(CHUNK)
            .map(|chunk| Self::of(chunk.iter().map(&project)))
            .reduce(|| Self::EMPTY, Self::combine)
    }

    /// The statistics of both domains together; see [`of`](Self::of) and
    /// [`of_slice`](Self::of_slice).
    fn combine(self, other: Self) -> Self {
        Self {
            max: self.max.max(other.max),
            grain: self.grain.min(other.grain),
            finite: self.finite && other.finite,
        }
    }

    /// The statistics of `values`.
    pub(crate) fn of(values: impl IntoIterator<Item = f32>) -> Self {
        values.into_iter().fold(Self::EMPTY, |domain, v| {
            if !v.is_finite() {
                Self {
                    finite: false,
                    ..domain
                }
            } else if v == 0.0 {
                domain
            } else {
                Self {
                    max: domain.max.max(v.abs()),
                    grain: domain.grain.min(grain(v)),
                    finite: domain.finite,
                }
            }
        })
    }

    /// Whether every value is finite (no NaN, no infinity).
    #[cfg_attr(
        not(all(target_os = "linux", feature = "cuda")),
        allow(
            dead_code,
            reason = "only the CUDA backend routes non-finite slices itself"
        )
    )]
    pub(crate) fn is_finite(&self) -> bool {
        self.finite
    }

    /// Whether every sum of at most `n` of the values, in any order and
    /// grouping, is computed exactly by both the CPU's `f64` sums and the
    /// GPU's integer accumulation of [`units`](Self::units), so that the
    /// two agree bit for bit: `n * max <= 2^53` grains, computed exactly.
    /// Non-finite values are never exact.
    pub(crate) fn sums_exact(&self, n: usize) -> bool {
        if !self.finite {
            return false;
        }
        if self.grain == i32::MAX {
            return true;
        }
        // `max / 2^grain` is an exact integer in `f64` (see the proof).
        let max_units = f64::from(self.max) * pow2(-self.grain);
        max_units <= MAX_UNITS as f64
            && (n as u128) * u128::from(max_units as u64) <= u128::from(MAX_UNITS)
    }

    /// The largest magnitude of the slice's values in grains:
    /// `max / 2^grain`, an exact integer (`0` for an empty or all-zero
    /// slice). The scatter kernel splits each grain count into a high and a
    /// low 16-bit piece, and a threadgroup's high-piece sum must stay inside
    /// an `i32`, so this bounds how many rows one threadgroup may scan.
    #[cfg_attr(
        not(all(target_os = "macos", feature = "metal")),
        allow(dead_code, reason = "only the Metal backend bounds its 32-bit pieces")
    )]
    pub(crate) fn max_units(&self) -> u64 {
        if self.grain == i32::MAX {
            return 0;
        }
        (f64::from(self.max) * pow2(-self.grain)) as u64
    }

    /// `x` in grains: the integer the GPU sums. Exact for a value of the
    /// slice whenever any sum of it is exact ([`sums_exact`](Self::sums_exact)
    /// holds for one value); otherwise (non-finite values, magnitudes past
    /// `2^63` grains) the saturating conversion keeps it defined, and such
    /// slices never reach the GPU.
    #[cfg_attr(
        not(all(target_os = "macos", feature = "metal")),
        allow(dead_code, reason = "the Metal backend stages grains on the host")
    )]
    pub(crate) fn units(&self, x: f32) -> i64 {
        (f64::from(x) * self.unit_scale()) as i64
    }

    /// A GPU sum of grains back in value space, exactly (see the proof).
    #[cfg_attr(
        not(all(target_os = "macos", feature = "metal")),
        allow(dead_code, reason = "the Metal backend scales sums on the host")
    )]
    pub(crate) fn value(&self, units: i64) -> f64 {
        units as f64 * self.value_scale()
    }

    /// The factor [`units`](Self::units) multiplies by before truncating:
    /// `2^-grain`, or `0` for an all-zero slice (so every value stages as
    /// `0`). A device staging the grains itself computes `units(x)` as
    /// `(f64::from(x) * unit_scale) as i64`.
    pub(crate) fn unit_scale(&self) -> f64 {
        if self.grain == i32::MAX {
            return 0.0;
        }
        pow2(-self.grain)
    }

    /// The factor [`value`](Self::value) multiplies by: `2^grain`, or `0`
    /// for an all-zero slice.
    pub(crate) fn value_scale(&self) -> f64 {
        if self.grain == i32::MAX {
            return 0.0;
        }
        pow2(self.grain)
    }
}

/// The exponent of the lowest set bit of a finite nonzero `f32`: the value
/// is an odd multiple of `2^grain`.
fn grain(v: f32) -> i32 {
    let bits = v.to_bits();
    let biased = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    let (significand, exponent) = if biased == 0 {
        (mantissa, -149)
    } else {
        (mantissa | 0x80_0000, biased - 150)
    };
    exponent + significand.trailing_zeros() as i32
}

/// `2^k` as an `f64`, exactly (`k` in the normal range).
fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model of the GPU path (`backend/metal.rs`): the values staged as
    /// grains, each slice of `slice_len` summed in `i64`, the slice totals
    /// merged in `i64`, and the total scaled back.
    fn gpu_sum(values: &[f32], slice_len: usize) -> f64 {
        let domain = SumDomain::of(values.iter().copied());
        let total: i64 = values
            .chunks(slice_len)
            .map(|slice| slice.iter().map(|&x| domain.units(x)).sum::<i64>())
            .sum();
        domain.value(total)
    }

    /// The CPU's sequential `f64` chain.
    fn cpu_sum(values: &[f32]) -> f64 {
        values.iter().fold(0.0f64, |s, &x| s + f64::from(x))
    }

    /// The magnitude in grains agrees with the bound `sums_exact` computes.
    #[test]
    fn max_units_matches_the_exactness_bound() {
        let values = [0.75f32, -3.0, 1.0 + f32::EPSILON];
        let domain = SumDomain::of(values.iter().copied());
        let max_units = domain.max_units();
        assert!(max_units >= 1);
        assert!(domain.sums_exact(1));
        assert!(domain.sums_exact((MAX_UNITS / max_units) as usize));
        assert!(!domain.sums_exact((MAX_UNITS / max_units) as usize + 1));
        assert_eq!(SumDomain::EMPTY.max_units(), 0);
        assert_eq!(SumDomain::of([0.0f32, -0.0].iter().copied()).max_units(), 0);
    }

    /// The parallel fold agrees with the serial one exactly, including for
    /// the values that make a slice inexact: the chunking must not change a
    /// domain (staging picks the grain and the bound from it).
    #[test]
    fn the_parallel_fold_matches_the_serial_one() {
        let mut values = vec![0.0, -0.0, -3.0, 0.75, 1.0 + f32::EPSILON, 2f32.powi(50)];
        values.extend((0..10_000).map(|i| (i as f32 * 0.37).sin() * 2f32.powi(i % 40 - 20)));
        for extra in [
            vec![],
            vec![f32::INFINITY, -1.0],
            vec![f32::NAN],
            vec![f32::from_bits(1)],
        ] {
            let all: Vec<f32> = values
                .iter()
                .copied()
                .chain(extra.iter().copied())
                .collect();
            assert_eq!(
                SumDomain::of_slice(&all, |v| *v),
                SumDomain::of(all.iter().copied())
            );
        }
    }

    #[test]
    fn grain_is_the_lowest_set_bit() {
        assert_eq!(grain(1.0), 0);
        assert_eq!(grain(-3.0), 0);
        assert_eq!(grain(0.75), -2);
        assert_eq!(grain(2f32.powi(50)), 50);
        assert_eq!(grain(8_388_609.0), 0); // 2^23 + 1
        assert_eq!(grain(f32::from_bits(1)), -149); // smallest subnormal
        assert_eq!(grain(f32::from_bits(0x40_0000)), -127); // 2^-127
        assert_eq!(grain(1.0 + f32::EPSILON), -23);
    }

    /// Staging and scaling back are exact at the extremes of the `f32`
    /// range: subnormals, the largest finite value, and mixed exponents.
    #[test]
    fn grains_round_trip_exactly() {
        for values in [
            vec![
                f32::from_bits(1),
                f32::from_bits(0x7F_FFFF),
                -f32::MIN_POSITIVE,
            ],
            vec![f32::MAX, -(2f32.powi(104))],
            vec![0.75, -3.0, 1.0 + f32::EPSILON],
        ] {
            let domain = SumDomain::of(values.iter().copied());
            assert!(domain.sums_exact(1));
            for &x in &values {
                assert_eq!(domain.value(domain.units(x)), f64::from(x), "{x:e}");
            }
        }
        let zeros = SumDomain::of([0.0f32, -0.0]);
        assert_eq!(zeros.units(0.0), 0);
        assert_eq!(zeros.value(0), 0.0);
    }

    /// The review's triggering input: the six values span 51 bits, so a
    /// node sums them exactly only up to `2^53 / 2^50 = 8` rows; the
    /// 8,192-row case goes to the CPU.
    #[test]
    fn wide_dynamic_range_bounds_the_row_count() {
        let six = [
            2f32.powi(50),
            2f32.powi(26),
            2f32.powi(23) + 1.0,
            -(2f32.powi(50)),
            -(2f32.powi(26)),
            -(2f32.powi(23)),
        ];
        let domain = SumDomain::of(six);
        assert!(domain.sums_exact(8));
        assert!(!domain.sums_exact(9));
        assert!(!domain.sums_exact(8192));
        assert_eq!(gpu_sum(&six, 4), cpu_sum(&six));
    }

    /// The boundary: `n * max` up to `2^53` grains is in, one more row is
    /// out; zeros, subnormals, and non-finite values.
    #[test]
    fn domain_boundary() {
        // Hessians of squared error: all 1.0, grain 2^0.
        let ones = SumDomain::of([1.0f32; 3]);
        assert!(ones.sums_exact(1 << 53));
        assert!(!ones.sums_exact((1 << 53) + 1));
        // An odd multiple of 2^-10 decides the grain; the max is 4 = 2^12
        // grains.
        let mixed = SumDomain::of([4.0, -0.5, 3.0 * 2f32.powi(-10), 0.0]);
        assert!(mixed.sums_exact(1 << 41));
        assert!(!mixed.sums_exact((1 << 41) + 1));
        // Subnormals are fine: the GPU never sees a float.
        assert!(SumDomain::of([f32::from_bits(1)]).sums_exact(1 << 53));
        // A single value past 2^53 grains is never exact.
        assert!(!SumDomain::of([2f32.powi(60), 1.0]).sums_exact(1));
        assert!(SumDomain::of([0.0f32, -0.0]).sums_exact(usize::MAX));
        assert!(SumDomain::EMPTY.sums_exact(usize::MAX));
        assert!(!SumDomain::of([1.0, f32::NAN]).sums_exact(1));
        assert!(!SumDomain::of([f32::INFINITY]).sums_exact(1));
    }

    /// Inside the domain the modeled GPU path equals the CPU chain bit for
    /// bit at the domain's edge: mostly-positive large values (partial
    /// sums near `2^53` grains, where the chain has no bit to spare) mixed
    /// with odd small ones, over several slice groupings.
    #[test]
    fn in_domain_sums_match_the_cpu_chain() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for case in 0..200 {
            let n = 4096 + (next() % 4096) as usize;
            // Largest magnitude 2^top with grain 2^0: `n * 2^top <= 2^53`.
            let top = 53 - (n as f64).log2().ceil() as i32;
            let values: Vec<f32> = (0..n)
                .map(|_| {
                    let r = next();
                    let sign = if (r >> 40) % 8 == 0 { -1.0 } else { 1.0 };
                    let v = match (r >> 1) % 4 {
                        // Large values with full 24-bit significands.
                        0 | 1 => ((r >> 8) % (1 << 24)) as f32 * 2f32.powi(top - 24),
                        // Odd small values.
                        2 => ((r >> 8) % 1000 * 2 + 1) as f32,
                        _ => 2f32.powi(top),
                    };
                    sign * v
                })
                .collect();
            let domain = SumDomain::of(values.iter().copied());
            assert!(domain.sums_exact(n), "case {case} must be in the domain");
            for slice_len in [1, 7, 128, 1024, n] {
                assert_eq!(
                    gpu_sum(&values, slice_len),
                    cpu_sum(&values),
                    "case {case}, slices of {slice_len}"
                );
            }
        }
    }

    /// Just past the bound the CPU chain itself rounds: `1 + k * 2^30`
    /// reaches `2^53 + 1` at `k = 2^23`, which `f64` rounds to `2^53`, while
    /// the exact (integer) sum keeps the 1. The domain refuses the node.
    #[test]
    fn the_cpu_chain_rounds_just_past_the_bound() {
        let k = (1 << 23) + 1;
        let values: Vec<f32> = std::iter::once(1.0)
            .chain(std::iter::repeat_n(2f32.powi(30), k))
            .collect();
        // The exact sum, `2^53 + 2^30 + 1`, is not an `f64`: compare in
        // integers.
        let exact = 1 + (k as i64) * (1 << 30);
        assert_eq!(cpu_sum(&values) as i64, exact - 1);
        assert!(!SumDomain::of(values.iter().copied()).sums_exact(values.len()));
    }
}
