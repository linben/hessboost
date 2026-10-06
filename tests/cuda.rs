//! CUDA backend integration tests (Linux, `cuda` feature).
//!
//! Tests that need a CUDA device skip when one is absent (CI runners have
//! no GPU) unless `HESSBOOST_REQUIRE_CUDA` is set, which turns every skip
//! into a failure: set it on a GPU machine so a broken setup cannot pass
//! vacuously. Kernel compilation is checked whenever NVRTC is loadable,
//! GPU or not; parameter-refusal tests always run.

#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use hessboost::backend::cuda::{self, CudaHistBackend, NodeCounts};
use hessboost::config::{
    BoosterKind, Dart, Device, GrowPolicy, LinearTree, MaxDeltaStep, Monotone, ProcessType,
    QuantizedGrad, Refresh, SamplingMethod, TrainingParamsBuilder,
};
use hessboost::internals::{CpuBackend, GHistIndex, HistCuts, HistogramBackend, zeroed};
use hessboost::objective::{GradPair, Multiclass, RegLoss};
use hessboost::prelude::*;

const CUDA: Device = Device::Cuda { ordinal: 0 };

/// Whether a CUDA device is usable, with the skip reason printed so a
/// vacuous pass is visible; panics instead under `HESSBOOST_REQUIRE_CUDA`.
fn device() -> bool {
    let Some(reason) = cuda::unavailable_reason() else {
        return true;
    };
    assert!(
        std::env::var_os("HESSBOOST_REQUIRE_CUDA").is_none(),
        "HESSBOOST_REQUIRE_CUDA is set but CUDA is unavailable: {reason}"
    );
    eprintln!("skipping cuda test: {reason}");
    false
}

/// The backend is either available or absent for a reason outside the
/// crate (no driver, no NVRTC, no device). A kernel compile or module load
/// failure is never an acceptable skip: without this guard, every
/// device-dependent test would pass vacuously while the backend is broken.
#[test]
fn backend_available_or_no_device() {
    if let Some(reason) = cuda::unavailable_reason() {
        assert!(
            ["libcuda not found", "libnvrtc not found", "no CUDA device"]
                .iter()
                .any(|expected| reason.starts_with(expected)),
            "the CUDA backend failed to initialize: {reason}"
        );
    }
}

/// The kernels compile for every architecture CUDA 12 and 13 both target,
/// with or without a GPU, whenever NVRTC is loadable.
#[test]
fn kernels_compile_for_supported_architectures() {
    for arch in ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90"] {
        match hessboost::internals::compile_kernels(arch) {
            Ok(bytes) => assert!(bytes > 0, "{arch}: empty CUBIN"),
            Err(error) if error.to_string().contains("libnvrtc not found") => {
                eprintln!("skipping kernel compile check: libnvrtc not found");
                return;
            }
            Err(error) => panic!("{error}"),
        }
    }
}

/// The unsupported `device = cuda` combinations are refused with an error,
/// never silently ignored.
#[test]
fn device_cuda_refuses_unsupported_combinations() {
    let base = TrainingParams::builder().device(CUDA).build().unwrap();
    let with = |change: fn(&mut TrainingParams)| {
        let mut params = base.clone();
        change(&mut params);
        params
    };
    let variants: Vec<(TrainingParams, &str)> = vec![
        (
            with(|p| p.tree_method = TreeMethod::Approx),
            "tree_method=approx",
        ),
        (
            with(|p| p.tree_method = TreeMethod::Exact),
            "tree_method=exact",
        ),
        (
            with(|p| p.quantized = Some(QuantizedGrad::default())),
            "use_quantized_grad",
        ),
        (
            with(|p| p.booster = BoosterKind::GbLinear),
            "booster=gblinear",
        ),
        (
            with(|p| p.process_type = ProcessType::Update(Refresh::default())),
            "process_type=update",
        ),
    ];
    for (params, name) in variants {
        assert_eq!(common::invalid_param(params.validate()), "device", "{name}");
    }
}

/// A dataset with missing values (every 13th cell) and a categorical first
/// column, as the Metal tests use.
fn dataset(n: usize, cols: usize, missing: bool) -> DMatrix {
    dataset_with(n, cols, missing, true)
}

/// [`dataset`] with the first column categorical or numeric.
fn dataset_with(n: usize, cols: usize, missing: bool, categorical: bool) -> DMatrix {
    let mut x = vec![0.0f32; n * cols];
    let mut y = vec![0.0f32; n];
    for r in 0..n {
        let mut target = 0.0;
        for f in 0..cols {
            let v = if f == 0 {
                ((r * 31 + f) % 5) as f32
            } else if missing && (r + f) % 13 == 0 {
                f32::NAN
            } else {
                (((r * 97 + f * 13) % 1000) as f32) * 0.001
            };
            x[r * cols + f] = v;
            if f > 0 && v.is_finite() {
                target += v * (f as f32);
            }
        }
        y[r] = target % 3.0;
    }
    let types: Vec<hessboost::data::FeatureType> = (0..cols)
        .map(|f| {
            if f == 0 && categorical {
                hessboost::data::FeatureType::Categorical
            } else {
                hessboost::data::FeatureType::Numerical
            }
        })
        .collect();
    DMatrix::from_dense_with_missing(&x, n, cols, f32::NAN)
        .unwrap()
        .with_feature_types(&types)
        .unwrap()
        .with_labels(&y)
        .unwrap()
}

/// The binned index of `n x cols` values from `value(row, feature)`
/// (`NaN` is missing).
fn index(n: usize, cols: usize, max_bin: usize, value: impl Fn(usize, usize) -> f32) -> GHistIndex {
    let x: Vec<f32> = (0..n * cols).map(|i| value(i / cols, i % cols)).collect();
    let data = DMatrix::from_dense_with_missing(&x, n, cols, f32::NAN).unwrap();
    let cuts = HistCuts::from_dmatrix(&data, max_bin);
    GHistIndex::from_dmatrix(&data, cuts)
}

/// The histogram of `rows` on `backend` after `prepare(gpair)`, as bits.
fn histogram(
    backend: &dyn HistogramBackend,
    index: &GHistIndex,
    rows: &[u32],
    gpair: &[GradPair],
) -> Vec<(u64, u64)> {
    let mut out = zeroed(index.total_bins());
    backend.prepare(index, gpair);
    backend.build(index, rows, gpair, &mut out);
    out.iter()
        .map(|s| (s.grad.to_bits(), s.hess.to_bits()))
        .collect()
}

/// Gradients whose sums are exact in grains (`k / 64` up to 16): every
/// node is exact.
fn exact_pairs(n: usize) -> Vec<GradPair> {
    (0..n)
        .map(|i| GradPair::new(((i * 7) % 1024) as f32 / 64.0 - 8.0, 1.0))
        .collect()
}

/// Gradients of magnitude 1 with one value of `2^-38`: `M = 2^38` grains,
/// so a chunk of up to 8,191 rows sums exactly but a node of more than
/// `2^15` rows does not.
fn chunk_exact_pairs(n: usize) -> Vec<GradPair> {
    (0..n)
        .map(|i| {
            let g = if i == 17 {
                2f32.powi(-38)
            } else if i % 3 == 0 {
                -1.0
            } else {
                1.0
            };
            GradPair::new(g, 1.0)
        })
        .collect()
}

/// Gradients spanning `1e-30` to `1e30` (and Hessians to `1e38`): no sum of
/// two of them is exact, so every node takes the `f64` chain path (or the
/// CPU's).
fn wide_pairs(n: usize) -> Vec<GradPair> {
    (0..n)
        .map(|i| {
            let e = (i * 13 % 61) as i32 - 30;
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            GradPair::new(
                sign * 10f32.powi(e) * (1.0 + (i % 7) as f32 / 8.0),
                10f32.powi((i % 39) as i32) * 0.9,
            )
        })
        .collect()
}

/// Every strategy's histograms equal the CPU backend's bit for bit: dense
/// and missing-value indexes, chains and chunked sums, inside and outside
/// the exactness domain, `u16` and `u32` bins. The node counts show each
/// strategy ran.
#[test]
fn histograms_match_cpu_for_every_strategy() {
    if !device() {
        return;
    }
    let mut seen = NodeCounts::default();
    let mut check = |index: &GHistIndex, rows: &[u32], gpair: &[GradPair], what: &str| {
        let gpu = CudaHistBackend::new(index, 0).unwrap();
        let cpu = histogram(&CpuBackend, index, rows, gpair);
        assert_eq!(histogram(&gpu, index, rows, gpair), cpu, "{what}");
        let counts = gpu.node_counts();
        seen.exact_nodes += counts.exact_nodes;
        seen.exact_chunk_nodes += counts.exact_chunk_nodes;
        seen.chain_nodes += counts.chain_nodes;
        seen.cpu_nodes += counts.cpu_nodes;
    };
    let value = |r: usize, f: usize| ((r * 2_654_435_761 + f * 97) % 1009) as f32 / 7.0;
    let with_missing = |r: usize, f: usize| {
        if (r + f).is_multiple_of(5) {
            f32::NAN
        } else {
            value(r, f)
        }
    };
    let sparse = |r: usize, f: usize| {
        if (r + f).is_multiple_of(4) {
            value(r, f)
        } else {
            f32::NAN
        }
    };

    // A dense index of 2^18 rows or fewer: every node is one chain.
    let small = index(60_000, 6, 256, value);
    let all: Vec<u32> = (0..60_000).collect();
    let thirds: Vec<u32> = (0..60_000).step_by(3).collect();
    let few: Vec<u32> = (0..60_000).step_by(11).take(5000).collect();
    check(&small, &all, &exact_pairs(60_000), "dense chain, exact");
    check(
        &small,
        &few,
        &wide_pairs(60_000),
        "dense small node, chains",
    );
    check(
        &small,
        &thirds,
        &wide_pairs(60_000),
        "dense large chain, cpu",
    );

    // A dense index above 2^18 rows: a row subset is chunked.
    let large = index(300_000, 4, 64, value);
    let half: Vec<u32> = (0..300_000).step_by(2).collect();
    check(&large, &half, &exact_pairs(300_000), "dense chunked, exact");
    check(
        &large,
        &half,
        &chunk_exact_pairs(300_000),
        "dense chunked, exact chunks",
    );
    check(&large, &half, &wide_pairs(300_000), "dense chunked, chains");

    // Missing values (a half-full index, and a CSR-only one): chunked.
    for (name, cells) in [
        ("missing", &with_missing as &dyn Fn(usize, usize) -> f32),
        ("csr", &sparse),
    ] {
        let idx = index(40_000, 7, 128, cells);
        let rows: Vec<u32> = (0..40_000).filter(|r| r % 7 != 3).collect();
        check(&idx, &rows, &exact_pairs(40_000), &format!("{name}, exact"));
        check(
            &idx,
            &rows,
            &chunk_exact_pairs(40_000),
            &format!("{name}, exact chunks"),
        );
        check(&idx, &rows, &wide_pairs(40_000), &format!("{name}, chains"));
        check(
            &idx,
            &rows[..3000],
            &wide_pairs(40_000),
            &format!("{name}, small chains"),
        );
    }

    // More than 65,536 bins: `u32` bins.
    let wide = index(20_000, 300, 256, |r, f| ((r * 31 + f * 7) % 20_000) as f32);
    let rows: Vec<u32> = (0..20_000).collect();
    check(&wide, &rows, &exact_pairs(20_000), "u32 bins, exact");
    check(
        &wide,
        &rows[..4000],
        &wide_pairs(20_000),
        "u32 bins, chains",
    );

    // A non-finite gradient anywhere in the slice: the CPU's NaN bits.
    let mut nan = exact_pairs(60_000);
    nan[11].grad = f32::NAN;
    nan[22].hess = f32::INFINITY;
    check(&small, &few, &nan, "non-finite, cpu");

    assert!(seen.exact_nodes > 0, "{seen:?}");
    assert!(seen.exact_chunk_nodes > 0, "{seen:?}");
    assert!(seen.chain_nodes > 0, "{seen:?}");
    assert!(seen.cpu_nodes > 0, "{seen:?}");
}

/// Inputs that do not fit the backend's device buffers never reach the
/// GPU: a gradient slice longer than the index and a row list longer than
/// the index give the CPU's histogram, and a row past the index is refused
/// by the CPU path's bounds check, exactly as the CPU backend refuses it.
#[test]
fn mismatched_inputs_match_the_cpu_backend() {
    if !device() {
        return;
    }
    let n = 10_000;
    let index = index(n, 1, 256, |r, _| (r % 5) as f32);
    let backend = CudaHistBackend::new(&index, 0).unwrap();
    let long: Vec<_> = (0..n + 1000)
        .map(|i| GradPair::new((i % 7) as f32 - 3.0, 1.0))
        .collect();
    let rows: Vec<u32> = (0..n as u32).collect();
    assert_eq!(
        histogram(&backend, &index, &rows, &long),
        histogram(&CpuBackend, &index, &rows, &long)
    );
    let gpair = &long[..n];
    let twice: Vec<u32> = rows.iter().chain(&rows).copied().collect();
    assert_eq!(
        histogram(&backend, &index, &twice, gpair),
        histogram(&CpuBackend, &index, &twice, gpair)
    );
    let past_end: Vec<u32> = (1..=n as u32).collect();
    let refused = |backend: &dyn HistogramBackend| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            histogram(backend, &index, &past_end, gpair)
        }))
        .is_err()
    };
    assert!(refused(&CpuBackend));
    assert!(refused(&backend));
}

/// `device = cuda` training reproduces single-threaded CPU training bit for
/// bit, tree for tree (the whole serialized model compares equal), across
/// the configurations the hist builder serves.
#[test]
fn device_cuda_training_matches_single_threaded_cpu() {
    if !device() {
        return;
    }
    let dense = dataset(40_000, 10, false);
    let missing = dataset(40_000, 10, true);
    let labelled = |f: fn(f32) -> f32| {
        let y: Vec<f32> = missing.labels().unwrap().iter().map(|&v| f(v)).collect();
        missing.clone().with_labels(&y).unwrap()
    };
    let binary = labelled(|v| f32::from(v >= 1.5));
    let classes = labelled(f32::floor);
    let numeric = dataset_with(40_000, 10, true, false);
    // Weights, and labels of exactly 1 for `scale_pos_weight` to reweight.
    let weights: Vec<f32> = (0..40_000).map(|i| 0.5 + (i % 7) as f32 * 0.25).collect();
    let ones: Vec<f32> = missing
        .labels()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 3 == 0 { 1.0 } else { v })
        .collect();
    let weighted = missing
        .clone()
        .with_labels(&ones)
        .unwrap()
        .with_weights(&weights)
        .unwrap();
    // Logistic rows: the weighted set's, binarized; an odd row count, whose
    // last rows run the host's scalar path; and margins beyond the host
    // vector kernel's range on some rows (those rounds grow on the host).
    let binary_labels: Vec<f32> = ones.iter().map(|&v| f32::from(v >= 1.0)).collect();
    let weighted_binary = weighted.clone().with_labels(&binary_labels).unwrap();
    let odd = dataset(40_003, 10, true);
    let odd_labels: Vec<f32> = odd
        .labels()
        .unwrap()
        .iter()
        .map(|&v| f32::from(v >= 1.5))
        .collect();
    let odd = odd.with_labels(&odd_labels).unwrap();
    let far: Vec<f32> = (0..40_000)
        .map(|i| if i % 997 == 0 { 85.0 } else { 0.0 })
        .collect();
    let far_margins = binary.clone().with_base_margin(&far).unwrap();
    let base = || {
        TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(6)
            .eta(0.3)
    };
    let monotone = vec![Monotone::None, Monotone::Increasing, Monotone::Decreasing];
    let configs: Vec<(&str, TrainingParamsBuilder, &DMatrix)> = vec![
        ("squared error", base(), &dense),
        (
            "weighted, scale_pos_weight",
            base().objective(Objective::SquaredError(RegLoss::new(2.5).unwrap())),
            &weighted,
        ),
        ("missing values", base(), &missing),
        (
            "logistic",
            base().objective(Objective::BinaryLogistic(RegLoss::default())),
            &binary,
        ),
        (
            "logistic, weighted, scale_pos_weight",
            base().objective(Objective::BinaryLogistic(RegLoss::new(2.5).unwrap())),
            &weighted_binary,
        ),
        (
            "logistic, scalar tail",
            base().objective(Objective::RegLogistic(RegLoss::default())),
            &odd,
        ),
        (
            "logitraw, margins past the vector range",
            base().objective(Objective::BinaryLogitRaw(RegLoss::default())),
            &far_margins,
        ),
        (
            "multiclass",
            base().objective(Objective::Softprob(Multiclass::new(3).unwrap())),
            &classes,
        ),
        ("subsample", base().subsample(0.7), &missing),
        (
            "gradient-based sampling",
            base()
                .subsample(0.5)
                .sampling_method(SamplingMethod::GradientBased),
            &missing,
        ),
        (
            "column sampling",
            base()
                .colsample_bytree(0.8)
                .colsample_bylevel(0.8)
                .colsample_bynode(0.8),
            &missing,
        ),
        (
            "alpha, gamma, max_delta_step",
            base()
                .alpha(0.5)
                .gamma(0.1)
                .max_delta_step(MaxDeltaStep::Bounded(0.5)),
            &missing,
        ),
        ("monotone", base().monotone_constraints(monotone), &dense),
        (
            "interaction",
            base().interaction_constraints(vec![vec![1, 2, 3], vec![4, 5]]),
            &dense,
        ),
        (
            "lossguide",
            base().grow_policy(GrowPolicy::LossGuide).max_leaves(31),
            &missing,
        ),
        (
            "symmetric",
            base().grow_policy(GrowPolicy::Symmetric),
            &numeric,
        ),
        (
            "dart",
            base().booster(BoosterKind::Dart(Dart::default())),
            &missing,
        ),
        (
            "posterior sampling",
            base().posterior_sampling(true),
            &missing,
        ),
        (
            "linear leaves",
            base().linear_tree(LinearTree::default()),
            &dense,
        ),
    ];
    for (name, builder, data) in configs {
        let cpu = builder.clone().build().unwrap();
        let gpu = builder.device(CUDA).build().unwrap();
        let train_one = |params: &TrainingParams| {
            common::with_threads(1, || train(params, data, 8).unwrap())
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(train_one(&cpu), train_one(&gpu), "{name}");
    }
}

/// On numeric data, depthwise trees grow resident: histograms stay on the
/// device, which subtracts siblings and scans every feature's splits. The
/// models still equal single-threaded CPU training bit for bit, across the
/// scorer's options (monotone bounds, `alpha`, `max_delta_step`,
/// `min_child_weight`), feature restrictions (interaction constraints,
/// column sampling), missing values, device-side rounds, and gradients so
/// large that scans score NaN (those nodes are searched on the host).
#[test]
fn device_cuda_resident_search_matches_single_threaded_cpu() {
    if !device() {
        return;
    }
    let dense = dataset_with(40_000, 10, false, false);
    let missing = dataset_with(40_000, 10, true, false);
    let binary_labels: Vec<f32> = missing
        .labels()
        .unwrap()
        .iter()
        .map(|&v| f32::from(v >= 1.5))
        .collect();
    let binary = missing.clone().with_labels(&binary_labels).unwrap();
    let weights: Vec<f32> = (0..40_000).map(|i| 0.5 + (i % 7) as f32 * 0.25).collect();
    let weighted = missing.clone().with_weights(&weights).unwrap();
    let huge_labels: Vec<f32> = dense.labels().unwrap().iter().map(|&v| v * 1e30).collect();
    let huge = dense.clone().with_labels(&huge_labels).unwrap();
    let base = || {
        TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(6)
            .eta(0.3)
    };
    let monotone = vec![Monotone::Increasing, Monotone::None, Monotone::Decreasing];
    let configs: Vec<(&str, TrainingParamsBuilder, &DMatrix)> = vec![
        ("dense", base(), &dense),
        ("missing values", base(), &missing),
        (
            "logistic",
            base().objective(Objective::BinaryLogistic(RegLoss::default())),
            &binary,
        ),
        ("weighted", base(), &weighted),
        ("monotone", base().monotone_constraints(monotone), &missing),
        (
            "alpha, lambda, max_delta_step, min_child_weight",
            base()
                .alpha(0.5)
                .lambda(2.0)
                .max_delta_step(MaxDeltaStep::Bounded(0.5))
                .min_child_weight(3.0),
            &missing,
        ),
        (
            "interaction",
            base().interaction_constraints(vec![vec![0, 1, 2], vec![3, 4]]),
            &missing,
        ),
        (
            "column sampling",
            base()
                .colsample_bytree(0.8)
                .colsample_bylevel(0.8)
                .colsample_bynode(0.8),
            &missing,
        ),
        ("depth 1", base().max_depth(1), &missing),
        ("NaN scans", base(), &huge),
    ];
    for (name, builder, data) in configs {
        let cpu = builder.clone().build().unwrap();
        let gpu = builder.device(CUDA).build().unwrap();
        let train_one = |params: &TrainingParams| {
            common::with_threads(1, || train(params, data, 8).unwrap())
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(train_one(&cpu), train_one(&gpu), "{name}");
    }
}

/// Past 2^18 rows a dense index's row subsets are summed in chunks, and the
/// partition spans many tiles per node: training there (dense and with
/// missing values, exact and inexact gradients, subsampled, depthwise and
/// loss-guided) still reproduces the CPU model bit for bit.
#[test]
fn device_cuda_training_matches_cpu_at_scale() {
    if !device() {
        return;
    }
    let dense = dataset_with(300_000, 8, false, false);
    let missing = dataset(300_000, 8, true);
    let y: Vec<f32> = missing
        .labels()
        .unwrap()
        .iter()
        .map(|&v| f32::from(v >= 1.5))
        .collect();
    let binary = missing.clone().with_labels(&y).unwrap();
    let base = || {
        TrainingParams::builder()
            .objective(Objective::SquaredError(RegLoss::default()))
            .tree_method(TreeMethod::Hist)
            .max_depth(7)
            .eta(0.3)
    };
    let logistic = || base().objective(Objective::BinaryLogistic(RegLoss::default()));
    let configs: Vec<(&str, TrainingParamsBuilder, &DMatrix)> = vec![
        ("dense", base(), &dense),
        ("dense subsample", base().subsample(0.6), &dense),
        ("missing", base(), &missing),
        ("logistic", logistic(), &binary),
        ("logistic subsample", logistic().subsample(0.5), &binary),
        (
            "lossguide",
            logistic().grow_policy(GrowPolicy::LossGuide).max_leaves(48),
            &binary,
        ),
    ];
    for (name, builder, data) in configs {
        let cpu = builder.clone().build().unwrap();
        let gpu = builder.device(CUDA).build().unwrap();
        let bytes = |params: &TrainingParams| {
            train(params, data, 4)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        };
        assert_eq!(bytes(&cpu), bytes(&gpu), "{name}");
    }
}

/// A `device = cuda` run repeats itself exactly, independent of the worker
/// count.
#[test]
fn device_cuda_training_is_deterministic() {
    if !device() {
        return;
    }
    let data = dataset(20_000, 9, true);
    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.3)
        .subsample(0.8)
        .device(CUDA)
        .build()
        .unwrap();
    let run = |threads| {
        common::with_threads(threads, || {
            train(&params, &data, 8)
                .unwrap()
                .encode(ModelFormat::Binary)
                .unwrap()
        })
    };
    assert_eq!(run(1), run(1));
    assert_eq!(run(1), run(4));
}

/// Training on a device ordinal that does not exist fails with a GPU error
/// instead of falling back silently.
#[test]
fn missing_device_ordinal_is_an_error() {
    if !device() {
        return;
    }
    let params = TrainingParams::builder()
        .tree_method(TreeMethod::Hist)
        .device(Device::Cuda { ordinal: 4096 })
        .build()
        .unwrap();
    let data = dataset(1_000, 3, false);
    assert!(matches!(
        train(&params, &data, 1),
        Err(hessboost::error::HessboostError::Gpu(_))
    ));
}
