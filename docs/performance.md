# Performance

hessboost trains on parallel histograms with runtime-detected SIMD kernels:
NEON on AArch64; AVX2+FMA (gradients, exp/sigmoid/softmax) and SSE2 (bin
search) on x86-64. Split search stays scalar; prediction walks monotone
integer keys branch-free. Everything else falls back to scalar Rust. Split
choices, histogram sums, and predictions match the scalar path exactly;
transcendental kernels stay within a few f32 ULPs of the scalar functions.
Below: CPU training against XGBoost, then the optimizations inside hessboost.

## XGBoost comparison

Measured on **Apple M3 Max** against [XGBoost 3.4.1](https://pypi.org/project/xgboost/3.4.1/),
the latest stable PyPI release checked on **2026-09-26 UTC**. Same dense `f32`
data and CPU `hist` settings on both sides: 100 rounds, depth 6, 256 bins,
`eta=0.1`, `lambda=1`. Times cover fresh training-matrix preparation plus
training; each is the median of six fits after warmup.

| Workload | Threads | hessboost | XGBoost 3.4.1 |
|---|---:|---:|---:|
| Regression, 100k × 30 | 1 | 0.412 s | 1.054 s |
| Regression, 100k × 30 | 4 | 0.158 s | 0.367 s |
| Regression, 100k × 30 | 16 | 0.203 s | 0.360 s |
| Regression, 50k × 128 | 1 | 1.057 s | 3.220 s |
| Regression, 50k × 128 | 4 | 0.357 s | 0.986 s |
| Regression, 50k × 128 | 16 | 0.326 s | 0.657 s |
| Binary, 100k × 30 | 1 | 0.402 s | 1.044 s |
| Binary, 100k × 30 | 4 | 0.156 s | 0.363 s |
| Binary, 100k × 30 | 16 | 0.199 s | 0.356 s |
| 4-class, 50k × 30 | 1 | 0.929 s | 2.505 s |
| 4-class, 50k × 30 | 4 | 0.299 s | 0.972 s |
| 4-class, 50k × 30 | 16 | 0.235 s | 1.188 s |

hessboost is faster in all 12 configurations: 2.6–3.0× single-threaded,
2.3–3.3× at four threads, and 1.8–2.0× at sixteen threads (5.0× on
multiclass).

![hessboost speedup over XGBoost 3.4.1 by workload and thread count](benchmarks/xgboost-speedup.svg)

![Median fit time by workload and thread count, log scale](benchmarks/xgboost-threads.svg)

Charts are rendered by
[`benchmarks/charts.gp`](benchmarks/charts.gp) from the table values in
[`benchmarks/xgboost.dat`](benchmarks/xgboost.dat) and
[`benchmarks/optimization.dat`](benchmarks/optimization.dat); regenerate with
`gnuplot -c docs/benchmarks/charts.gp` after updating a data file.

### CPU scheduling

XGBoost spreads histogram work over nodes and row blocks, split search over
nodes and features, and its updater also emits final row positions — the
places to look when a faster kernel barely moves total training time. See the
upstream
[histogram builder](https://github.com/dmlc/xgboost/blob/v3.4.1/src/tree/hist/histogram.h),
[split evaluator](https://github.com/dmlc/xgboost/blob/v3.4.1/src/tree/hist/evaluate_splits.h),
and [hist updater](https://github.com/dmlc/xgboost/blob/v3.4.1/src/tree/updater_quantile_hist.cc).

hessboost overlaps independent depthwise nodes, sizes histogram tasks by
node, and shares the worker pool with data preparation and margin updates.
Gains depend on tree shape, feature count, sampling, and worker count; more
workers isn't always faster.

### Model quality

Same held-out rows on both sides, drawn separately from the training rows.
Scores below are the single-thread fits; lower is better. Synthetic tasks
with identical hyperparameters — a comparability check, not a general
quality claim.

| Workload | Held-out rows | Metric | hessboost | XGBoost 3.4.1 |
|---|---:|---|---:|---:|
| Regression, 100k × 30 | 20,000 | rmse | 0.060965 | 0.060965 |
| Regression, 50k × 128 | 10,000 | rmse | 0.064210 | 0.064210 |
| Binary, 100k × 30 | 20,000 | logloss | 0.516273 | 0.516273 |
| 4-class, 50k × 30 | 10,000 | mlogloss | 0.150027 | 0.150027 |

From the M3 Max run itself (2026-09-26, both engines): hessboost matches
XGBoost within 1e-9 on every workload, so the timing gaps are speed, not
fit quality.

### Workloads and method

- Regression: `2*x0 - 3*x1² + 0.5*x2 + x3*x4` plus N(0, 0.05²). The
  128-feature case adds irrelevant features.
- Binary: Bernoulli draws with log-odds `4*(x0-0.5) - 3*(x1-0.5) + 2*(x2-0.5)`.
- Four-class: argmax of `3*xi - x((i+1) mod 4)` plus N(0, 0.1²); four trees
  per round.

Features uniform on `[0, 1)`; NumPy and training seed 1234. Held-out sets are
one fifth the training rows. Both sides: depthwise growth,
`base_score=0.5`, `alpha=0`, `gamma=0`, `min_child_weight=1`, full
row/column sampling, no early stopping or eval callbacks in the timer.

XGBoost uses its macOS ARM64 PyPI wheel (OpenMP) with `QuantileDMatrix`
built in the timer; hessboost builds a fresh `DMatrix` in the timer and bins
during training. Both read identical binary data beforehand. Excluded from
timing: test-matrix prep, file I/O, startup, prediction, scoring, teardown.

Runtime: macOS 27.0, Rust 1.98.1 / LLVM 22.1.8, uv Python 3.14.5, NumPy
2.5.2, XGBoost 3.4.1. Rust release profile, no extra `RUSTFLAGS`; XGBoost
native build reports Clang 15 + OpenMP. Compiled first, engines run
sequentially. Interactive workstation with unrelated CPU activity — treat
small gaps as near parity.

Batches run XGBoost/hessboost/hessboost/XGBoost per workload and thread
count: one warmup fit discarded, three recorded, per batch; the table takes
the median of all six fits per engine. This CPU and these synthetic
workloads only — no GPU or cross-platform claim. Full per-fit samples,
scores, build info, and source hashes land in the output directory
(`/tmp/hessboost-xgb-full` for this run; reruns write a new one).

Reproduce from the repository root, using a new output directory:

```sh
cargo build --release --example bench_compare
uv run --with xgboost==3.4.1 --with numpy==2.5.2 python scripts/bench_xgb.py \
  --hessboost target/release/examples/bench_compare \
  --output /tmp/hessboost-xgb --threads 1 4 16
```

Workload selection and a quick harness check are in the
[script docs](../scripts/README.md#xgboost-comparison).

## Optimization benchmarks

Optimized vs scalar baseline, same source and compiler. **Apple M3 Max** (16
physical cores), macOS 26.6.2, **Rust 1.98.1 / LLVM 22.1.8**, 2026-09-14
UTC. Both builds: `opt-level=3`, thin LTO, one codegen unit, no
`RUSTFLAGS`; `RAYON_NUM_THREADS` fixed per comparison. Compiled and tested
before timing; binaries run sequentially on a live workstation, alternating
order to cut (not kill) timing bias.

Each value is the mean of two Criterion run medians in
baseline/optimized/optimized/baseline order (0.5 s warmup, 1 s measurement,
20 samples, 10k bootstraps; Criterion extends as needed). **Less time** is
`100 × (1 − optimized / baseline)`. This machine and these workloads only.

### Full training

All cases train 50 depth-six trees on 50,000 × 20. Data creation is outside
the timer; training covers quantile prep, gradients, tree building, and
training-prediction updates.

| Workload | Threads | Baseline (ms) | Optimized (ms) | Less time |
|---|---:|---:|---:|---:|
| Regression, 256 bins | 1 | 398.027 | 110.772 | 72.2% |
| Regression, 256 bins | 4 | 381.738 | 54.300 | 85.8% |
| Regression, L1 = 1 | 1 | 392.963 | 107.278 | 72.7% |
| Regression, L1 = 1 | 4 | 379.340 | 53.641 | 85.9% |
| Regression, 16 bins | 1 | 231.349 | 82.752 | 64.2% |
| Regression, 16 bins | 4 | 220.740 | 43.935 | 80.1% |
| Binary classification | 1 | 408.616 | 114.892 | 71.9% |
| Binary classification | 4 | 396.398 | 57.560 | 85.5% |

![Full-training time cut vs scalar baseline](benchmarks/training-optimization.svg)

### Single histogram tree

Cuts, bins, and gradients are prepared outside the timer; the cases isolate
tree construction (sampling, histograms, partitioning, split evaluation).

| Workload | Threads | Baseline (ms) | Optimized (ms) | Less time |
|---|---:|---:|---:|---:|
| Depth 1 | 1 | 0.855 | 0.361 | 57.8% |
| Depth 1 | 4 | 0.618 | 0.220 | 64.3% |
| Depth 6 | 1 | 6.866 | 2.155 | 68.6% |
| Depth 6 | 4 | 6.726 | 0.908 | 86.5% |
| Depth 10 | 1 | 49.609 | 12.373 | 75.1% |
| Depth 10 | 4 | 49.458 | 3.926 | 92.1% |
| 128 features | 1 | 24.854 | 5.742 | 76.9% |
| 128 features | 4 | 24.886 | 2.310 | 90.7% |
| Missing values | 1 | 9.409 | 5.938 | 36.9% |
| Missing values | 4 | 9.287 | 2.438 | 73.7% |
| Monotone constraint | 1 | 9.193 | 4.096 | 55.4% |
| Monotone constraint | 4 | 9.008 | 1.475 | 83.6% |
| Loss-guide growth | 1 | 6.843 | 3.077 | 55.0% |
| Loss-guide growth | 4 | 6.680 | 2.924 | 56.2% |

Depth cases: 50,000 × 20. Wide: 10,000 × 128 at depth six. Missing,
monotone, and loss-guide reuse the depth-six set; missing fills 2 of every
11 entries, monotone constrains the first feature upward. All 256 bins;
loss-guide caps at 64 leaves.

![Histogram tree-build time cut vs scalar baseline](benchmarks/tree-optimization.svg)

### Numerical kernels

Pointwise cases run one million predictions (multiclass: `1M / classes`
rows, ~1M outputs). Transforms include the copy into the reusable output
buffer; metrics include final normalization. Prepared inputs, single
thread, no tree training.

![Objective gradient time cut vs scalar baseline](benchmarks/gradient-optimization.svg)

![Prediction transform time cut vs scalar baseline](benchmarks/transform-optimization.svg)

![Metric time cut vs scalar baseline](benchmarks/metric-optimization.svg)

| Workload | Threads | Baseline (ms) | Optimized (ms) | Less time |
|---|---:|---:|---:|---:|
| Logistic gradient | 1 | 2.041 | 0.809 | 60.4% |
| Logistic gradient, weighted | 1 | 2.038 | 0.811 | 60.2% |
| Poisson gradient | 1 | 2.608 | 0.932 | 64.2% |
| Gamma gradient | 1 | 1.380 | 0.579 | 58.0% |
| Tweedie gradient | 1 | 2.630 | 1.056 | 59.8% |
| Softmax gradient, 2 classes | 1 | 4.210 | 0.833 | 80.2% |
| Softmax gradient, 3 classes | 1 | 5.890 | 0.810 | 86.2% |
| Softmax gradient, 4 classes | 1 | 3.574 | 0.901 | 74.8% |
| Softmax gradient, 8 classes | 1 | 2.675 | 1.309 | 51.1% |
| Softmax gradient, 32 classes | 1 | 2.176 | 1.012 | 53.5% |
| Softmax gradient, 128 classes | 1 | 2.188 | 0.948 | 56.6% |
| Sigmoid transform | 1 | 1.448 | 0.573 | 60.4% |
| Exponential transform | 1 | 1.280 | 0.473 | 63.1% |
| Softmax transform, 2 classes | 1 | 2.339 | 0.564 | 75.9% |
| Softmax transform, 3 classes | 1 | 2.378 | 0.543 | 77.2% |
| Softmax transform, 4 classes | 1 | 1.997 | 0.581 | 70.9% |
| Softmax transform, 8 classes | 1 | 1.896 | 0.712 | 62.4% |
| Softmax transform, 32 classes | 1 | 1.877 | 0.581 | 69.1% |
| Softmax transform, 128 classes | 1 | 1.927 | 0.589 | 69.4% |
| RMSE, weighted | 1 | 0.776 | 0.268 | 65.4% |
| MAE, weighted | 1 | 0.768 | 0.268 | 65.1% |
| Binary error, weighted | 1 | 1.338 | 0.204 | 84.8% |
| Log loss, weighted | 1 | 4.930 | 2.738 | 44.5% |
| Poisson NLL, weighted | 1 | 2.598 | 1.545 | 40.5% |
| Gamma NLL, weighted | 1 | 2.633 | 1.575 | 40.2% |
| Tweedie NLL, weighted | 1 | 13.797 | 4.521 | 67.2% |
| Multiclass log loss, 32 classes, weighted | 1 | 0.102 | 0.062 | 39.5% |
| Multiclass error, 32 classes, weighted | 1 | 1.616 | 0.244 | 84.9% |

The suite also covers unweighted metrics, more class counts, and
histogram-accumulation controls (scalar loop and scheduling).

### Quantized-gradient training (opt-in)

`use_quantized_grad` (LightGBM-style quantized training, not XGBoost
behavior) packs each bin's two `f64` sums into one integer — 32-bit up to
`32767 / Q` rows, else 64-bit. Oversized nodes accumulate row runs in an
L1-resident 32-bit scratch histogram first. Rows read as one `i32` instead
of an 8-byte gradient pair. Split scoring reads dequantized sums, paying one
extra pass per split to dequantize both child histograms.

**AWS Neoverse-V3** (192 cores, Linux 6.12), **Rust 1.98.1**, `opt-level=3`,
thin LTO, one codegen unit, 2026-09-23 UTC. Criterion medians (2 s warmup,
6 s measurement) on a busy host — treat gaps under ~3% as noise. Q = 4,
stochastic rounding, no leaf renewal. The full-precision column is the same
cases on the same machine and run, plus a 1M × 50 case.

| Workload | Threads | Full precision (ms) | Quantized (ms) | Speedup |
|---|---:|---:|---:|---:|
| Tree, depth 6, 50k × 20 | 1 | 5.935 | 5.982 | 0.99× |
| Tree, depth 6, 50k × 20 | 16 | 1.387 | 1.314 | 1.06× |
| Tree, depth 10, 50k × 20 | 1 | 64.10 | 65.70 | 0.98× |
| Tree, depth 10, 50k × 20 | 16 | 5.197 | 5.195 | 1.00× |
| Tree, 128 features, 10k rows | 1 | 26.35 | 26.77 | 0.98× |
| Tree, 128 features, 10k rows | 16 | 5.527 | 5.298 | 1.04× |
| Tree, missing values | 1 | 11.50 | 11.32 | 1.02× |
| Tree, missing values | 16 | 2.875 | 2.825 | 1.02× |
| Tree, depth 8, 1M × 50 | 1 | 274.9 | 181.7 | **1.51×** |
| Tree, depth 8, 1M × 50 | 16 | 29.81 | 16.09 | **1.85×** |
| Training, regression, 50 rounds | 1 | 308.8 | 318.5 | 0.97× |
| Training, regression, 50 rounds | 16 | 77.22 | 71.92 | 1.07× |
| Training, binary, 50 rounds | 1 | 309.4 | 323.7 | 0.96× |
| Training, binary, 50 rounds | 16 | 80.24 | 75.01 | 1.07× |

Only worth it when accumulation dominates the build — the 1M-row case, where
it is about half the profile. On 50k rows and 128 features, gain evaluation
(identical code in both modes) dominates: the integer histograms save about
a third of the smaller accumulation share, and the quantization plus
per-split dequantization passes give most of it back. Net: 3–4% slower
single-threaded training on 50k rows, ~7% faster at 16 threads. The integer
loops are scalar (per-node widths, bit-identical serial/parallel); no SIMD
path to keep in sync.

### Split search, histogram, and scheduling changes

Sixteen fixed workloads, median of five fits after warmup (fresh `DMatrix`
in the timer; prediction/SHAP time only the call), all at once on
NUMA-local CPU sets. Output bits hashed before and after — none changed at
any thread count; XGBoost parity passes.

192-core **AWS Neoverse-V3** (Rust 1.98.1, release), 2026-09-24 UTC, before
and after the changes under [Implementation](#implementation). Busy host
(load ~90), so single values vary up to ~10%.

| Case | Threads | Before (ms) | After (ms) |
|---|---:|---:|---:|
| Regression 100k × 30, 100 rounds | 1 | 1247 | 668 |
| Regression 100k × 30, 100 rounds | 4 | 425 | 283 |
| Regression 100k × 30, 100 rounds | 16 | 261 | 185 |
| Regression 50k × 128, 100 rounds | 1 | 3931 | 1582 |
| Regression 50k × 128, 100 rounds | 16 | 758 | 436 |
| Binary 100k × 30, 100 rounds | 1 | 1186 | 650 |
| Binary 100k × 30, 100 rounds | 16 | 260 | 172 |
| 4-class 50k × 30, 100 rounds | 1 | 3312 | 1490 |
| 4-class 50k × 30, 100 rounds | 16 | 805 | 219 |
| Loss-guide (64 leaves) 100k × 30 | 16 | 1597 | 312 |
| Binary 100k × 30, 20% missing | 16 | 729 | 158 |
| Exact 50k × 20, 20 rounds | 4 | 2906 | 361 |
| Regression 1M × 50, depth 8, 20 rounds | 48 | 441 | 361 |
| Predict 100k × 30, 100 trees | 1 | 132 | 132 |
| Predict 100k × 30, 100 trees | 16 | 8.64 | 9.72 |
| SHAP contributions, 2k rows | 16 | 27.9 | 27.6 |
| **Geometric mean** | | **501.8** | **247.5** |

Prediction and SHAP run unchanged code; deltas are host noise.

### Hot-loop code generation

Rewritten where codegen, not the algorithm, was the limit:

- **Prediction:** lockstep walk steps each lane with `cmp` + `cinc`
  (`simd::step_if_greater`) — LLVM had compiled the plain select to a
  branch random rows mispredict ~25% of the time. Keys form branch-free,
  one row at a time, then scatter to lane slots.
- **SHAP:** eight rows walk each tree in lockstep, overlapping their `f32`
  chains. Edge terms and child bases are NEON kernels (LLVM had scalarized
  the divisions); subtrees return weighted values by value; paths and rows
  sit feature-major; return edges sum two rows at a time.
- **Exact:** per-row scan keeps node stats in registers; precomputed
  incumbent test screens each candidate in ~10 flops; one loop per scan
  direction with direction as a constant; leaf rows update margins directly.
- **Histogram:** paired feature scans overlap prefix-sum chains; non-negative
  Hessians take extremes at endpoints; dense accumulation tiles 1,024 rows;
  register-resident partition predicates with raw-pointer writes; ~2,048
  candidates per split-scan task; children evaluated side by side from
  4,096 rows.
- **Data prep:** fully present 64-row blocks binned feature by feature;
  interleaved branch-free sketch merges; bare-key radix sort for
  unit-weight queues.

**Apple M3 Max** (macOS 27.0, Rust 1.98.1), 2026-09-26 UTC, before
(`35b2cd9`) vs after (`c94b0c8`): eight workloads at 1 and 8 threads,
median of fifteen fits each (fresh `DMatrix` in the timer for training),
builds interleaved baseline/optimized/optimized/baseline. Output hashes
identical; values are the mean of the two run medians.

| Case | Threads | Before (ms) | After (ms) | Speedup |
|---|---:|---:|---:|---:|
| Regression 100k × 30, depth 6, 100 rounds | 1 | 480.2 | 408.0 | 1.18× |
| Binary 100k × 30, depth 6, 100 rounds | 1 | 512.8 | 443.2 | 1.16× |
| 4-class 50k × 30, depth 6, 50 rounds | 1 | 493.2 | 398.7 | 1.24× |
| Regression 50k × 128, depth 6, 50 rounds | 1 | 744.1 | 624.0 | 1.19× |
| Loss-guide (64 leaves) 100k × 30, 50 rounds | 1 | 389.7 | 327.5 | 1.19× |
| Exact 20k × 20, depth 6, 30 rounds | 1 | 582.9 | 246.4 | 2.37× |
| Predict 100k × 30, 100 trees | 1 | 25.57 | 23.62 | 1.08× |
| SHAP contributions, 2k × 20, 100 trees | 1 | 319.3 | 170.3 | 1.87× |
| Regression 100k × 30, depth 6, 100 rounds | 8 | 183.8 | 153.8 | 1.19× |
| Binary 100k × 30, depth 6, 100 rounds | 8 | 189.5 | 157.2 | 1.21× |
| 4-class 50k × 30, depth 6, 50 rounds | 8 | 131.5 | 102.3 | 1.29× |
| Regression 50k × 128, depth 6, 50 rounds | 8 | 194.3 | 162.3 | 1.20× |
| Loss-guide (64 leaves) 100k × 30, 50 rounds | 8 | 146.2 | 122.0 | 1.20× |
| Exact 20k × 20, depth 6, 30 rounds | 8 | 204.9 | 79.1 | 2.59× |
| Predict 100k × 30, 100 trees | 8 | 4.00 | 3.88 | 1.03× |
| SHAP contributions, 2k × 20, 100 trees | 8 | 55.6 | 27.4 | 2.03× |
| **Geometric mean (per-thread)** | 1 | **337.3** | **248.6** | **1.36×** |
| **Geometric mean (per-thread)** | 8 | **93.7** | **67.1** | **1.40×** |
| **Geometric mean (all 16)** | | **177.7** | **129.1** | **1.38×** |

Hist training ~1.2× (Neoverse-V3 measured 1.15–1.30× on the same shapes),
exact ~2.4–2.6× (was 1.94–2.73×), SHAP ~1.9–2.0× (was 2.00–2.05×).
Prediction here is the `predict_100k_x30_100trees_depth6` bench shape
(depthwise model, generic lockstep walk, ~1.03–1.08×) — unchanged within
binary-layout noise per [Prediction and explanations](#prediction-and-explanations).
The old 4× prediction row replayed the `step_if_greater` microbenchmark
(random rows mispredicting the old branch).

### Model serialization

The writer pre-sizes each payload and the whole container before filling;
the compact encoder emits bytes not bits and binary-searches the sorted
dictionaries. Parsed `CompactModel` keeps one padded byte copy, not two.
The XGBoost importer reads node arrays in place (no per-tree `f64` copies)
and keeps category segments as ranges. Output bytes unchanged.

192-core **AWS Neoverse-V3** (Rust 1.98.1, bench profile), 2026-09-25 UTC,
`scripts/compare_benchmarks.py` (baseline/optimized/optimized/baseline, 20
samples, mean of run medians), `RAYON_NUM_THREADS=1`, busy host (load
~100). Models: 100 trees on 20,000 × 20, depth six and loss-guide (255
leaves). Compact and UBJSON cases from a throwaway harness of the same
shape; the rest is `model_io_100trees_depth6`.

| Case | Threads | Before (ms) | After (ms) | Less time |
|---|---:|---:|---:|---:|
| `to_compact_bytes`, depth 6 | 1 | 1.921 | 0.779 | 59.4% |
| `to_compact_bytes`, loss-guide | 1 | 8.909 | 4.406 | 50.5% |
| `from_xgboost_ubjson`, depth 6 | 1 | 2.450 | 2.191 | 10.6% |

`CompactModel::from_bytes`, compact prediction, native `to_bytes` /
`from_bytes`, and the XGBoost exporters are unchanged within host noise
(under 2%; `from_xgboost_json` +3.2%, at the edge of it). The native writer
is zstd-bound. A shared-buffer writer variant measured 14% slower and was
dropped.

### Data preparation

CSR rows bin straight from the matrix (no per-row entry copy); quantile and
sketch radix sorts share one implementation with per-worker reusable
buckets; sketch merges swap buffers; CSR cut counts come from the column
view; text loaders reuse one line buffer. Cuts and bins unchanged.

Same host/method as serialization, vs the previous commit.
`data_prep_100k_x30` (20 samples); loaders parse 1M × 20 from memory vs
`BufRead::lines` (10 samples).

| Case | Threads | Before (ms) | After (ms) | Less time |
|---|---:|---:|---:|---:|
| `GHistIndex::from_dmatrix`, CSR | 1 | 11.323 | 10.685 | 5.6% |
| `GHistIndex::from_dmatrix`, CSR | 16 | 0.851 | 0.795 | 6.6% |
| `HistCuts::from_dmatrix`, dense | 1 | 54.765 | 54.051 | 1.3% |
| `HistCuts::from_dmatrix`, dense | 16 | 4.353 | 4.282 | 1.6% |
| `read_csv`, 1M rows | 1 | 454.9 | 436.4 | 4.1% |

Dense cuts repeat in both runs of each pair (+1.3%/+1.3% at 1 thread,
+1.4%/+1.8% at 16). CSR cut construction, dense binning, and `read_libsvm`
are within noise. Dropped as slower: one-pass CSR categorical validation
(10% slower when categorical columns lead), a reused `(sum, count)` buffer
for target stats (17% slower).

### Training rounds

Under `approx` with a non-constant Hessian, every tree of an output's
forest (`num_parallel_tree > 1`) now shares one row/gradient sample and one
per-round cut weighting per output — the gradient index builds once per
round, not once per tree, and the weighted sketch reads Hessians in place.
Models unchanged.

Same host/method vs the previous commit: 50,000 × 20, 20 rounds, depth 6,
`binary:logistic`, `approx`, forest of 4, throwaway harness.

| Case | Threads | Before (ms) | After (ms) | Less time |
|---|---:|---:|---:|---:|
| `approx` logistic forest of 4 | 1 | 2896.9 | 981.7 | 66.1% |
| `approx` logistic forest of 4 | 16 | 223.6 | 107.5 | 52.0% |

With `linear_tree` and no zero-weight rows, the builder's final partition
feeds the linear-leaf fit (and the margin update through the leaf models)
instead of re-routing every row through the new tree. Every value was
sketched, so each leaf sees the same rows in the same order; models
unchanged. Zero weights keep the old routing (a zero-weight row can sit past
the last cut, where builder and tree disagree).

| Case | Threads | Before (ms) | After (ms) | Less time |
|---|---:|---:|---:|---:|
| `train_variants_50k_x20_20rounds/linear_tree` | 1 | 223.8 | 139.0 | 37.9% |
| `train_variants_50k_x20_20rounds/linear_tree` | 16 | 35.0 | 27.8 | 20.5% |

The other `train_variants_50k_x20_20rounds` cases (DART, CSR, `approx`
forests, eval sets, vector leaves, one-tree-per-output) are within noise.
Dropped as slower or flat: a dense CSR margin scratch row (+12% at 1
thread), extending the cached prediction layout per tree (+2% everywhere),
in-place gradient sampling (no gain).

### Prediction and explanations

Single-row compact walk keeps ≤128-feature keys on the stack; compact
layout builds in one breadth-first pass; gblinear margins parallelize over
rows in order; vector-leaf SHAP reads each output's tree from the leaf
vectors instead of cloning per output. Predictions and attributions
unchanged.

Same host/method vs the previous commit, throwaway harness:
`predict_100k_x30_100trees_depth6` shape (100 depth-six trees, 30 features;
gblinear on 100,000 × 30).

| Case | Threads | Before (ms) | After (ms) | Less time |
|---|---:|---:|---:|---:|
| `predict_margin`, 1 row | 1 | 0.0009 | 0.0008 | 8.3% |
| `predict_margin`, 8 rows | 1 | 0.0065 | 0.0059 | 8.4% |
| `predict_leaf`, 8 rows | 1 | 0.0045 | 0.0040 | 10.1% |
| `predict_contribs`, vector leaves, 1 row | 1 | 0.405 | 0.391 | 3.5% |
| gblinear `predict_margin`, 100k rows | 16 | 6.581 | 0.717 | 89.1% |

Batch prediction, SHAP, single-threaded gblinear, and conformal calibration
are unchanged — same hot loops, deltas within ±4% binary-layout noise (two
builds of the old code differing only in unrelated training code measured
140.0 vs 132.8 ms). A shared block-tail walk for generic and symmetric
kernels was dropped (+4.6% quantile, +2.2% symmetric).

### Categorical and sparse partitions

Categorical splits and sub-half-full indexes used to partition serially
through per-row lookups; both now use the branch-free partition loop
([Implementation](#implementation)) keyed off the bin, so trees are
unchanged (unit-tested vs per-row routing).

192-core **AWS Neoverse-V3** (Rust 1.98.1, bench profile), 2026-09-25 UTC,
`scripts/compare_benchmarks.py` (medians,
baseline/optimized/optimized/baseline; 10 samples, 100 for
`hist_tree_build/missing`), before/after on the benchmark-coverage commit,
busy host.

| Case | Threads | Before (ms) | After (ms) |
|---|---:|---:|---:|
| `train_variants_50k_x20_20rounds/categorical` | 1 | 104.67 | 90.70 |
| `train_variants_50k_x20_20rounds/categorical` | 16 | 23.02 | 20.00 |
| `train_variants_50k_x20_20rounds/csr` | 1 | 112.90 | 95.40 |
| `train_variants_50k_x20_20rounds/csr` | 16 | 45.08 | 28.64 |
| `train_50k_x20_50rounds/Hist` (dense, numeric) | 1 | 169.57 | 168.58 |
| `train_50k_x20_50rounds/Hist` (dense, numeric) | 16 | 43.33 | 43.76 |
| `hist_tree_build/missing` | 1 | 3.83 | 3.86 |
| `hist_tree_build/missing` | 16 | 0.82 | 0.81 |
| 4,000-category feature, 50k rows, depth 10, 20 rounds | 1 | 985.2 | 155.7 |
| 4,000-category feature, 50k rows, depth 10, 20 rounds | 16 | 903.3 | 67.4 |

Dense numeric and half-full cases run unchanged code — host noise. The
4,000-category throwaway (one categorical + three numeric features,
`max_bin = 4096`, single runs): each split builds a per-bin left-set table
(binary search per category) instead of testing every bin against the set.

### Objective gradients

Count gradients share the logistic/softmax fixed row chunks; LambdaRank
parallelizes over query groups (disjoint row writes); AFT evaluates each
row's endpoint densities once; expectile gradients rebuild each row's
expectiles once, not once per output. Arithmetic and order unchanged
(bit-for-bit vs serial, unit-tested).

Same host/method (`objective_gradient`, `objective_gradient_other`):

| Case | Threads | Before (ms) | After (ms) |
|---|---:|---:|---:|
| `poisson_unweighted_1m` | 1 | 1.606 | 1.604 |
| `poisson_unweighted_1m` | 16 | 1.599 | 0.120 |
| `gamma_unweighted_1m` | 16 | 1.041 | 0.082 |
| `tweedie_unweighted_1m` | 16 | 1.858 | 0.136 |
| `rank_ndcg_100k_groups100` | 1 | 21.607 | 21.319 |
| `rank_ndcg_100k_groups100` | 16 | 21.602 | 1.387 |
| `rank_map_100k_groups100` | 16 | 19.557 | 1.279 |
| `rank_pairwise_100k_groups100` | 16 | 17.288 | 1.117 |
| `aft_normal_1m` | 1 | 60.156 | 32.679 |
| `aft_normal_1m` | 16 | 3.784 | 2.063 |
| `expectile_a3_1m_outputs` | 1 | 19.340 | 11.091 |
| `expectile_a3_1m_outputs` | 16 | 1.446 | 0.837 |

Single-threaded Poisson/LambdaRank run the serial code — host noise.

### Evaluation metrics

AUC, AUCPR, ungrouped ranking metrics, and `cox-nloglik` sort once via
rayon's stable parallel merge sort (same order as serial); NDCG/MAP/`pre@k`
score groups in parallel and reduce in group order; `aft-nloglik` and
`interval-regression-accuracy` parallelize row values and sum in row order.
Values unchanged (bit-for-bit 4-vs-1 threads, unit-tested); serial paths
keep their loops. Same host/method (`eval_metric_other`):

| Case | Threads | Before (ms) | After (ms) |
|---|---:|---:|---:|
| `auc_100k` | 1 | 1.831 | 1.864 |
| `auc_100k` | 16 | 1.833 | 0.833 |
| `aucpr_100k` | 1 | 2.068 | 2.098 |
| `aucpr_100k` | 16 | 2.084 | 1.218 |
| `auc_100k_k3_matrix` | 1 | 5.809 | 5.674 |
| `auc_100k_k3_matrix` | 16 | 5.582 | 2.775 |
| `ndcg_100k_groups100` | 1 | 2.390 | 2.386 |
| `ndcg_100k_groups100` | 16 | 2.388 | 0.180 |
| `map_100k_groups100` | 1 | 0.968 | 0.954 |
| `map_100k_groups100` | 16 | 0.970 | 0.093 |
| `pre@5_100k_groups100` | 1 | 0.945 | 0.940 |
| `pre@5_100k_groups100` | 16 | 0.945 | 0.171 |
| `aft-nloglik_100k` | 1 | 2.841 | 2.529 |
| `aft-nloglik_100k` | 16 | 2.841 | 0.241 |
| `interval-regression-accuracy_100k` | 1 | 0.285 | 0.283 |
| `interval-regression-accuracy_100k` | 16 | 0.285 | 0.108 |
| `cox-nloglik_100k` | 1 | 1.509 | 1.533 |
| `cox-nloglik_100k` | 16 | 1.506 | 1.115 |

Single-threaded paths run the serial code — host noise.

## Implementation

The private `simd` module owns dispatch and kernels. AArch64 checks NEON
once per process, x86-64 checks AVX2+FMA once; the result is cached.
Everything else is scalar Rust, no target flags needed. Dispatch checks
lengths before entering an unsafe kernel; vector traffic stays in complete
blocks. Scalar formulas serve fallbacks, exceptional blocks, and tails.

| Operation | NEON path |
|---|---|
| Logistic, Poisson, Gamma, Tweedie gradients | Four `f32` predictions per block, with weighted and unweighted inputs |
| Sigmoid and exponential transforms | Four `f32` predictions per block |
| Softmax gradients and transforms, 2–4 classes | Four rows at a time using interleaved loads and stores |
| Softmax gradients and transforms, 8+ classes | Vector blocks within each row |
| RMSE, MAE, binary error, log loss, count metrics | `f64` reductions with optional weights |
| Multiclass log loss | Gathered label probabilities with `f64` logarithms |
| Multiclass error, 8+ classes | Vector row maxima |

Minimum 16 elements for most kernels; 5–7-class softmax stays scalar. The
`f32` exp is range reduction plus a degree-seven polynomial on finite
`[-80, 80]`, `f32::exp` elsewhere; softmax subtracts the row max and bails
on nonfinite rows or margin spread above 80. Estrin evaluation for
gradients and narrow softmax, Horner for wide in-place softmax. Metric logs
and Tweedie exps stay `f64`.

Depthwise growth expands nodes and draws child feature samples in traversal
order, then partitions, histograms, and evaluates independent nodes in
parallel. Wide nodes also split numeric scans into parallel feature chunks,
merged in feature order under XGBoost's tie rule — the sequential result.
Loss-guide keeps priority-queue order (ids, stats, sampler draws), building
the next-best children ahead of turn in parallel unless per-level/per-node
column sampling is on. One iteration's trees (per class, or a
`num_parallel_tree` forest) grow concurrently after slot-ordered RNG draws.
Exact scans a level's features in parallel the same way.

Split scoring batches per feature: prefix sums in bin order, then an `f32`
closed form `G · (G / (H + λ))` per child that vectorizes. Only candidates
within `2^-16` relative of the best approximation re-score exactly, in
order; monotone/`alpha`/`max_delta_step`/reuse-penalty/non-finite
configs score everything exactly, still batched. Exact skips candidates a
division-free bound rules out. `tree::builder::tests` checks both against
sequential search.

The root sweeps the column-major bin copy two features at a time, one
writer per bin. Other subsets up to 2^18 rows gather per feature pair from
the same copy — no partial histograms, rows ascending per bin. Larger nodes
(8,192+ rows, sparse or big) split into `n / 4,096` fixed blocks, each
summed from zero and added in block order, one wave per worker count — sums
depend on rows, never thread count (serial sums the same blocks). Row
sweeps prefetch four bins before storing. At 8 threads the fixed blocks
cost ~5% on the 50k-row `missing` build, nothing measurable at 1M rows.

Leaves at `max_depth` skip histograms and split search. Under full row
sampling, training keeps their final partitions (depthwise, loss-guide,
exact) and adds leaf values straight into cached margins. Sampled and eval
sets update disjoint rows in parallel, skipping inputs too small to
parallelize. Stats, monotone bounds, and sampler draws preserved.

Cuts sort per feature; rows bin in parallel chunks. Unweighted sketch
queues radix-sort (whole-number weights sum identically in any order).
Large dense inputs validate and copy in parallel. Collection preserves cut
layout, row order, missing handling, categorical bins, and 16/32-bit bin
choice. Half-full-or-denser sparse indexes keep a column-major copy with a
missing sentinel for single-column streaming (categorical splits stream the
same columns through a per-bin left-set table); sparser ones scan stored
bins inline, in parallel row-order chunks for large nodes. Architecture-independent.

### Prediction

`BoostedModel` derives a prediction layout lazily on first use, drops it
when a tree is appended, never serializes it. Each tree is renumbered
breadth-first into a 16-byte-node arena with adjacent children; each
numeric split becomes one ordered compare that is false for missing (left-
missing stores the next-lower threshold; right-missing mirrors children and
stores a negated threshold plus sign mask; leaves self-loop). One step: node
load, feature load, XOR, compare, add — no data-dependent branch. Sixteen
rows walk in lockstep for the tree depth; batches under sixteen rows walk
sixteen trees in lockstep instead. 256-row blocks run in parallel, each
block's sixteen-row groups stored feature-major (`[group][feature][lane]`);
trees sum in order, bit-identical to sequential. Non-`NaN` sentinels scatter
straight in; >4,096 sparse columns use per-lookup access. Categorical
splits and depth-16+ trees take an early-exit walk. ~6 IPC on Neoverse V3,
issue-bound.

#### Symmetric trees

Symmetric trees (same split at every node of a level: `grow_policy =
symmetric`, or any imported tree of that shape) skip the walk for full
sixteen-row groups. Each level stores one `(slot, key)` compare plus a
`2^depth` leaf table indexed by the level-outcome bit pattern (root most
significant); collapsed subtrees fill their slots. One vectorizable 16-lane
compare per level, no dependent load chain. Same compares and leaves as the
generic walk, so margins and leaf ids are bit-identical (unit-tested); tails,
tiny batches, depth < 2 or > 16, and oversized tables keep the generic walk.

`cargo bench --bench training -- predict_100k` on 100,000 × 30 with 100
depth-6 trees (`eta = 0.1`, rest default). Criterion medians on 192-core
**Neoverse V3** (Rust 1.98.1, 2026-09-23), `RAYON_NUM_THREADS` fixed. The
middle column disables the table path (one-line local change) to isolate
the kernel:

| Threads | Depthwise model (ms) | Symmetric model, generic walk (ms) | Symmetric model, bit pattern (ms) | Speedup, same model |
|---:|---:|---:|---:|---:|
| 1 | 132.13 | 99.64 | 13.37 | 7.5× |
| 16 | 8.63 | 6.57 | 1.03 | 6.4× |
| 192 | 1.66 | 1.42 | 0.70 | 2.0× |

Shared row loading and scheduling dominate at full width. The 10–20× figure
vs per-node traversal is an informal spot check from another machine, not a
recorded artifact.

SHAP is XGBoost 3.4's QuadratureTreeSHAP: one recursive walk per tree with
an 8-lane `f32` quadrature basis, each return edge's contribution read off
its subtree's return. Contributions `O(L · D)` per tree and row, interactions
`O(L · D²)` (`L` leaves, `D` depth); classic path-dependent TreeSHAP needs
`O(L · D²)` for contributions and a conditioned walk per feature for
interactions. Precomputed nodes hold both branch weights; only the tree's own
split features are cleared and accumulated; rows run in parallel. Spot
check on 192-core Neoverse V3 (hist, 20 features, 100 trees on 20,000 rows;
mean of 3–5 calls, not a Criterion artifact):

| Workload | Threads | Classic TreeSHAP | QuadratureTreeSHAP | Speedup |
|---|---:|---:|---:|---:|
| contributions, depth 6, 2,000 rows | 192 | 4.2 ms | 3.8 ms | 1.1× |
| contributions, depth 10, 2,000 rows | 192 | 46.1 ms | 29.9 ms | 1.5× |
| interactions, depth 6, 200 rows | 192 | 19.7 ms | 1.9 ms | 10.6× |
| interactions, depth 10, 200 rows | 192 | 285 ms | 14.4 ms | 19.8× |
| contributions, depth 6, 2,000 rows | 1 | 486 ms | 385 ms | 1.3× |
| contributions, depth 10, 2,000 rows | 1 | 7.49 s | 4.47 s | 1.7× |
| interactions, depth 6, 200 rows | 1 | 1.90 s | 77 ms | 24.6× |
| interactions, depth 10, 200 rows | 1 | 29.6 s | 1.14 s | 26.1× |

## Numerical behavior and validation

Objectives output `f32`; metrics and histogram stats accumulate `f64`. SIMD
rounding means cross-architecture results aren't bit-identical. Same
inputs, params, seed, and execution config still trains deterministically.

SHAP mirrors XGBoost 3.4.2's arithmetic: quadrature rule built in `f64`,
stored `f32`; recurrence and accumulations `f32` in XGBoost's order
(categorical children in XGBoost's orientation); per-tree expected value
summed in `f64`, rounded once. aarch64 XGBoost fuses `a * b + c`, x86_64
wheels don't; hessboost matches per target, so imported models reproduce
XGBoost's contributions and interactions bit for bit on the parity fixtures
(aarch64 Linux). Unfused stays within 2e-5 of fused there. The 8-point rule
is exact up to seven distinct path features; longer paths are XGBoost's own
quadrature approximation.

Kernels are tested against scalar formulas (short inputs, tails, weights,
saturation, NaN/inf, softmax ties). Gradient tolerance `1e-6 * max(1,
|reference|)`; metrics `1e-12`–`3e-12` relative on finite fixtures; split
tests check order and the sequential gain epsilon. Test tolerances, not
general error bounds.

## Metal GPU (macOS)

The `metal` feature adds a native Metal backend (`src/backend/metal.rs` has
the design and determinism contract). **Apple M4 Max** (40-core GPU, 14 CPU
cores, 16 threads, macOS 27.0, Rust 1.98.1, 2026-09-28):

| Workload | CPU | Metal | Speedup |
|---|---|---|---|
| predict, 500k rows × 30 features, 200 depth-8 trees | 25.7 ms | 8.0 ms | **3.2×** |
| train, 200k × 30, depth 8, 50 rounds | 177 ms | 285 ms | 0.62× |

Histogram builds, by node size (the crate's `metal_histogram_build`
benches, 4 CPU threads against the 40-core GPU):

| Rows × 30 features | CPU | Metal | Speedup |
|---|---|---|---|
| 4M | 6.62 ms | 5.75 ms | **1.15×** |
| 1M | 1.81 ms | 2.11 ms | 0.86× |
| 100k | 0.30 ms | 0.54 ms | 0.56× |

GPU **prediction** is the win: independent per-row walks over an L2-resident
compact forest, with each call's fixed row upload amortized over larger
batches and ensembles (larger models widen the 3.2×). The kernel is bound by
the cache lines a warp's scattered node loads touch — 32 rows walk different
nodes at every level — so fewer bytes per node is the lever that pays: a model
that fits it is uploaded in an 8-byte-per-node encoding (a threshold value
plus a packed feature/child word) instead of the arena's 16 bytes, and the
walk then compares floats rather than rebuilding the CPU's monotone key.
Categorical splits, vector leaves, and multi-output models keep 16 bytes per
node. Row materialization is at the memory system's limit either way, so a
call is pipelined in row blocks and one block's upload runs while the GPU
walks another.

GPU **histograms** (`device = metal`) scatter a node's rows into a
threadgroup-shared histogram through 32-bit atomics (`hist_scatter_u16`),
each 64-bit grain count split into exact high and low pieces that the piece
merge rejoins. Integer addition is order-free, so inside the exactness bound
this reproduces the CPU's `f64` sums bit for bit; a node whose grain counts
would overflow the pieces runs the register-bin kernels instead. A build's
per-row cost is sublinear in the node's rows (a fixed gather-and-merge cost per
node amortizes over more of them), so the GPU crosses the CPU between 1M and
4M rows per node: `device = metal` pays for the large nodes of a big dataset
and trails the CPU below that. A node of at least `CPU_ROWS` (8,192) rows is
offered to the GPU; smaller nodes, non-finite gradients, and data outside the
exactness bound run on the CPU. Which threshold minimizes *training* wall time
is machine- and workload-dependent — a CPU thread has slack while the GPU
waits — so the constant is a conservative default, not a portable optimum.

Run the Metal benches on a Mac with a Metal device:

```sh
cargo bench --features metal --bench training -- metal
cargo run --release --features metal --example metal
```

Hosted macOS CI has no Metal device: those tests skip there.

## CUDA GPU (Linux)

The `cuda` feature adds an NVIDIA backend (`src/backend/cuda/mod.rs` has the
design and exactness contract); every model it trains is byte-identical to
single-threaded CPU training. **AWS g6e.4xlarge**: NVIDIA L40S (driver
595.91.07, ECC on), AMD EPYC 7R13 (16 vCPUs), Rust 1.98.1, 2026-10-06.

Against XGBoost 3.4.1's PyPI wheel (CUDA 13.3) with `device=cuda`, through
`scripts/bench_xgb.py --device cuda --threads 16` (100 rounds, depth 6, 256
bins, `eta=0.1`; fresh matrix preparation plus training, median of six fits):

| Workload | hessboost (CUDA) | XGBoost 3.4.1 (CUDA) | Held-out score (both) |
|---|---:|---:|---|
| Regression, 1M × 30 | 0.501 s | 0.652 s | RMSE 0.057865 |
| Binary, 1M × 30 | 0.500 s | 0.652 s | logloss 0.511879 |
| Regression, 10M × 30 | 4.087 s | 5.750 s | RMSE 0.056821 |
| Binary, 10M × 30 | 3.920 s | 5.766 s | logloss 0.512848 |

The held-out scores agree to eight digits. At 10M rows about 2.7 s of
hessboost's time is matrix preparation (the quantile sketch and binning, on
the CPU, so that the cuts stay the CPU's); the rounds themselves take about
17 ms at depth 8 (CPU: 350 ms; XGBoost on the GPU: 25 ms).

The crate's `cuda` benches (`cargo bench --features cuda --bench training --
cuda`):

| Bench | CPU (16 threads) | CUDA | Speedup |
|---|---:|---:|---:|
| Histogram build, 1M rows × 30 | 4.37 ms | 0.825 ms | **5.3×** |
| Histogram build, 10M rows × 30 | 37.7 ms | 7.62 ms | **4.9×** |
| Train, 1M × 30, depth 8, 20 rounds | 837 ms | 334 ms | **2.5×** |

Where the time goes: each depthwise level is one partition, one batch of
histogram builds (shared-memory integer histograms in 512-thread blocks,
three resident per SM), the sibling subtractions, and one split-scan launch
(one warp per node and feature: lane 0 forms the CPU's sequential prefix
chain, the lanes score candidates in parallel, which is where the `f64`
divisions run at Ada's 1/64 rate). The host merges per-feature winners and
decides the tree; for squared error and logistic objectives the gradients
and margins never leave the GPU.

## Reproduce the measurements

Benchmarks live in [`benches/training.rs`](../benches/training.rs). Run the
suite on the current checkout with:

```sh
RAYON_NUM_THREADS=1 cargo bench --bench training
```

The quantized-gradient rows use the `*_quantized` cases of `hist_tree_build`
and the `Hist_quantized` / `quantized` training cases.

Unrecorded groups exist for `scripts/compare_benchmarks.py` coverage:
`objective_gradient_other`, `eval_metric_other`, the rest of
`train_variants_50k_x20_20rounds`, `predict_csr_100trees_depth6`,
`model_io_100trees_depth6`, `data_prep_100k_x30`, and `histogram_build`'s
`sparse` case.

Charts regenerate from the `.dat` files with:

```sh
gnuplot -c docs/benchmarks/charts.gp
```

Build-to-build comparison uses `scripts/compare_benchmarks.py`; see
[Development scripts](../scripts/README.md) for it, the quality fixtures,
and the timing harness.
