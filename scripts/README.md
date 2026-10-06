# Development scripts

## Parity fixtures

`gen_fixtures.py` trains **real XGBoost 3.4.2** (single thread) on deterministic
synthetic datasets, one case per supported feature (tree methods, missing values,
constraints, every objective — including `reg:logistic`, the `reg:linear`
alias, the alpha-list objectives with one and three alphas (list-valued
`quantile_alpha` / `expectile_alpha` params), and the survival objectives
with censored and tied times — sample weights, `scale_pos_weight` on every
objective XGBoost reads it in (`binary:logistic`, `binary:logitraw`,
`reg:logistic`, `reg:squarederror`, `reg:linear`, `reg:gamma`: the
`*_spw*` cases, whose regression labels are exactly 1 on about a third of
the rows), ranking groups, gblinear, DART, intercept estimation, row/column
sampling including `sampling_method=gradient_based` and DMatrix
`feature_weights`, multi-target
label matrices, per-round metric oracles for `rmsle`, `mape`, `mphe`,
`pre`/`pre@k`, the survival metrics and each objective's default metric,
continued training, `process_type=update`, `num_parallel_tree` forests,
`iteration_range` and slicing), and writes each case to
`../fixtures/<name>.json`: data, the exact `xgb.train` parameter dict,
XGBoost's test-set predictions (transformed, raw margin, SHAP contributions on
the first 50 rows, and for the cases in `INTERACTION_CASES` SHAP interaction
values on the first 5 rows: numeric, missing-value, categorical, multiclass,
DART, and the multi-output layouts, i.e. label matrix, alpha list,
parallel-tree forests, and vector-leaf trees) and the saved model JSON, plus
the model's UBJSON encoding (`save_raw("ubj")`) as `../fixtures/<name>.ubj`,
named by the `xgb_model_ubj` field. Case names say what they vary: `exact_` /
`approx_` / `hist_` tree methods, `_d<k>` the depth, and `nobs_` cases drop
`base_score` so the intercept is estimated. `weighted` cases carry
`weights`, ranking cases `group_sizes` / `test_group_sizes`, and categorical
cases (two integer-coded categorical columns) `feature_types`;
`categorical_missing_reg_d6` (6 and 80 categories with missing values, so the
forward and backward partition scans differ and `max_cat_threshold` binds)
and `categorical_onehot_reg_d6` (3 and 2 categories, one-hot splits) cover
the categorical split search. `n_targets`
gives the label columns: the `multi_*` cases (3-target `reg:squarederror`
on hist and exact and with `scale_pos_weight`, multi-label
`binary:logistic` with and without `scale_pos_weight`, weighted 2-target
`reg:pseudohubererror` and
`reg:absoluteerror`) store `y_train`/`y_test` row-major `[row][target]`, and
their predictions, margins, and contributions carry the target axis. The
`mot_*` cases train `multi_strategy=multi_output_tree` (vector-leaf trees):
3-target `reg:squarederror` depthwise, lossguide, with missing values,
regularized (`gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`,
`max_delta_step`), monotone, interaction-constrained and categorical;
multi-label `binary:logistic`; `multi:softprob` / `multi:softmax`; weighted
`reg:pseudohubererror`; `reg:quantileerror` / `reg:expectileerror` alpha
lists and a weighted `reg:absoluteerror` label matrix; a `num_parallel_tree=3`
forest with iteration ranges and slices; continued `multi:softprob` training
with ranges; plus quality-tier subsampling and DART. Their contributions
(and, for several, interactions) exercise vector-leaf QuadratureTreeSHAP.
The `forest_*`, `rf_*`, and `boosted_rf_*` cases cover `num_parallel_tree`
on scalar, multiclass, label-matrix, and alpha-list models.

It also writes `../fixtures/cuts/<name>.json`, XGBoost's `hist` and `approx`
quantile cuts (`DMatrix.get_quantile_cut`) for a set of matrices (uniform,
few distinct values, normal, missing values, sparse weights; `approx` cases
with squared-error and logistic round-0 Hessians).

`tests/parity.rs` runs these checks per case:

1. **Train parity** - train on the fixture data, compare `predict(x_test)` with
   XGBoost's predictions.
2. **Import parity** - `BoostedModel::decode` (`ModelFormat::XgboostJson`) on the embedded model,
   compare predictions, raw margins, SHAP contributions, and (where recorded)
   SHAP interaction values; decoding the `.ubj` sidecar (`XgboostUbjson`) must
   yield the identical model (column `ubj`).
3. **Export parity** - encode `XgboostJson` (`<name>.model.json`),
   `XgboostUbjson` (`<name>.model.ubj`) and hessboost's predictions to
   `../fixtures/exports/`; `check_exports.py` reloads each model in XGBoost and
   compares predictions, and for UBJSON also requires every array to use the
   same container form (typed element marker or generic) as XGBoost's own
   `save_raw("ubj")` of the loaded model.

4. **Feature extras** (column `extra`: largest delta / number of checks, `-`
   when the case has none), from optional fixture fields:
   - `continuation` (`first_rounds`, `xgb_model_initial`): XGBoost trained
     `first_rounds`, saved the model, and continued to `num_round` with
     `xgb_model=`. Train parity then runs `train` + `Trainer::init_model`
     the same way, and the imported initial model is continued and compared
     too.
   - `refresh` (`n_rows`, `y`, `rounds`, `refresh_leaf`, `xgb_pred`):
     `process_type=update` + `updater=refresh` of the final model on the first
     `n_rows` training rows relabelled `y`; hessboost refreshes the imported
     model (and, for `exact`-tier cases, its own) with `Trainer::init_model`.
   - `ranges` / `range_contribs` / `slices`: `iteration_range=(begin, end)`
     margins (hessboost's `predict_margin(data, begin..end)`), prefix-range
     contributions and leaf indices on the contribution rows, and
     `booster[begin:end:step]` margins (`slice(begin..end, step)`). Checked
     on the imported model (leaf ids included) and, for `exact`-tier cases,
     the trained model.

`quantile_cuts_match_xgboost` compares the cut oracles bit-for-bit with
`HistCuts::from_dmatrix` (`hist`) and `HistCuts::from_dmatrix_weighted` with
the round-0 Hessians (`approx`).

Cases are tiered. `exact` cases are pointwise: max |delta| within `tol.train`
(1e-4; 1e-5 for probabilities), `tol.import` (1e-5), `tol.contribs` (1e-4) and
`tol.interactions` (1e-4).
`quality` cases are RNG-driven (`subsample`, `sampling_method=gradient_based`,
`colsample_*` with or without `feature_weights`, DART, random forests);
training uses a regression RMSE <= 1.08x XGBoost band (accuracy >= XGBoost -
0.02 for classification) while import/export remain pointwise. A case's
optional `feature_weights` array (one weight per column) is set on the
training DMatrix on both sides. The `trainonly`-tier gblinear case validates
training pointwise; its unsupported XGBoost-JSON import/export path is
visibly reported as `n/a`/`skipped` and required to return `ModelFormat` on
import. Unknown XGBoost
parameters fail the test.

Optional fixture fields extend the schema for metadata beyond plain labels:

- `label_lower_bound` / `label_upper_bound` (training rows) and
  `test_label_lower_bound` / `test_label_upper_bound` (test rows): survival
  label bounds, attached with `DMatrix::with_label_bounds`. JSON has no
  infinity, so `+inf`/`-inf` are the strings `"inf"`/`"-inf"`; bounds are never
  NaN. A case whose objective reads the bounds only (`survival:aft`) has empty
  `y_train`/`y_test`, and no labels are attached.
- `test_weights`: per-row test-set weights (constant within a query group;
  XGBoost receives one weight per group for ranking cases).
- `xgb_evals`: `{metric: [value per round]}`, XGBoost's `evals_result()` on the
  labeled test set (labels, bounds, groups, `test_weights`) for cases built
  with the `evals` option. The metrics are the params' `eval_metric` list, or
  the objective's default metric when it is absent. The Rust side trains with
  a `Trainer` watching the same set and requires the same metric names and,
  every round, `|hessboost - xgboost| <= tol.evals * max(1, |xgboost|)`
  (`tol.evals` = 1e-5; column `evals`, `-` for cases without oracles).

```sh
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/gen_fixtures.py
cargo nextest run --test parity --release --run-ignored only --no-capture
uv run --with-requirements scripts/requirements-xgboost.txt python scripts/check_exports.py
```

Fixtures are not checked in; CI regenerates them (`.github/workflows/ci.yml`,
job `parity`). The generator refuses any XGBoost version other than 3.4.2, pinned in
`requirements-xgboost.txt` (a source build: 3.4.2 has no PyPI wheel; CI
caches the built wheel in uv's cache).

Fixture generators share deterministic seed, dtype-explicit JSON-float, dense-array, and compact JSON-writing helpers in `_fixture_common.py`; their case matrices and data distributions remain separate. `gen_lightgbm_fixtures.py --test-data` is parsed as a command-line option and regenerates the checked-in small models under `tests/data/`.

## LightGBM import parity

`gen_lightgbm_fixtures.py` trains **real LightGBM 4.7.0**
(`requirements-lightgbm.txt`; single thread, `deterministic`) and writes each
case to `../fixtures/lightgbm/<name>.txt` (the `save_model` text model) and
`<name>.json` (test rows, LightGBM's raw scores, predictions,
`pred_contrib` SHAP values, `pred_leaf` indices, raw scores of a half-length
slice, and what the import must do). The test rows include, for up to 40
numeric splits, the threshold rounded to `f32` and its two `f32`
neighbours, so the `<=`-on-doubles conversion is checked at its edge. Cases:
numeric, `NaN` and zero missing values, `zero_as_missing` (a small model
whose splits all map, and one that must be refused), categorical splits
with multi-word bitsets, multiclass and one-vs-all, `lambdarank` and
`rank_xendcg`, `linear_tree` (with categorical routing), every other
objective, DART and GOSS; refusals: `sigmoid` other than 1, `reg_sqrt`,
`cross_entropy_lambda`, and random forests (`average_output`).

`tests/lightgbm_parity.rs` imports each model and compares pointwise within
`1e-5` relative (`max(1, |LightGBM|)`), checks leaf indices, the slice, and
native binary/JSON and XGBoost JSON round trips; refused cases must fail
with a `ModelFormat` error naming the reason. `--test-data` instead
rewrites the small checked-in `tests/data/lightgbm-4.7.0-*` models and
their `*.expected.json` predictions.

```sh
uv run --with-requirements scripts/requirements-lightgbm.txt python scripts/gen_lightgbm_fixtures.py
cargo nextest run --test lightgbm_parity --release --run-ignored only --no-capture
```

## Criterion comparisons

`compare_benchmarks.py` runs two compiled Criterion executables in
baseline/optimized/optimized/baseline order with a fixed Rayon thread count.
Each run gets its own result directory. The script writes `comparison.json`
with medians and confidence intervals, and `samples.json.gz` with raw Criterion
samples and console output. It requires only the Python standard library.

```sh
python3 scripts/compare_benchmarks.py \
  --baseline /path/to/baseline-training-bench \
  --optimized /path/to/optimized-training-bench \
  --output /tmp/hessboost-comparison \
  --threads 1 --filter 'hist_tree_build|train_50k_x20_50rounds/Hist'
```

The output directory must not already exist. Build both executables with the
same benchmark source, lockfile, compiler, and release settings before running
the comparison. See [Performance](../docs/performance.md) for recorded results,
workload definitions, and complete reproduction commands.

## XGBoost comparison

`bench_xgb.py` generates shared training and held-out datasets, then benchmarks
XGBoost and the compiled `bench_compare` Rust example. Both engines read the
same little-endian `f32` bytes. Each timed fit constructs a fresh training
matrix and trains the model. File I/O, test-data preparation, evaluation, model
destruction, and process startup are outside the timer. XGBoost uses
`QuantileDMatrix` with CPU `hist`. hessboost builds its `DMatrix` and performs
binning during training.

The default suite covers regression, wide regression, binary classification,
and four-class classification, with 100 boosting rounds at 1, 4, and 16
threads. Each comparison runs in XGBoost/hessboost/hessboost/XGBoost order. Every
batch discards one warmup fit and records three fits. The report uses the
median of all six measurements for each engine. Held-out RMSE or log loss
checks model quality alongside timing.

```sh
cargo build --release --example bench_compare
uv run --with xgboost==3.4.1 --with numpy==2.5.2 python scripts/bench_xgb.py \
  --hessboost target/release/examples/bench_compare \
  --output /tmp/hessboost-xgb --threads 1 4 16
```

Use a new output directory for each comparison. It contains the shared binary
datasets and `comparison.json`, including every timing sample, held-out score,
training parameters, native XGBoost build information, package versions, and
source, executable, and dataset hashes. Build before timing and avoid running
other benchmarks or compiler jobs concurrently.

For a quick harness check, add `--rows 512 --rounds 3 --repeats 1`. Use
`--workloads regression` to select one dataset or `--threads 1` for a
single-thread comparison. Remove the package version constraints to benchmark
the latest releases available through `uv`. The output records the versions
actually used. To compare against the parity pin instead, replace the two
`--with` options with `--with-requirements scripts/requirements-xgboost.txt`
(XGBoost 3.4.2, a source build).

`--device cuda` trains both engines on the GPU (Linux with an NVIDIA GPU):
build the example with `--features cuda`, and use the PyPI `xgboost` wheel
(built with CUDA 13, so the driver must support CUDA 13), since the parity
pin is a CPU-only source build. The report records the GPU name, driver, and
ECC mode; the warmup fit absorbs CUDA context creation and hessboost's
kernel compilation. XGBoost's GPU sketch and gradient quantization give a
different model than its CPU `hist`, so compare the held-out scores too.

```sh
cargo build --release --features cuda --example bench_compare
uv run --with xgboost==3.4.1 --with numpy==2.5.2 python scripts/bench_xgb.py \
  --hessboost target/release/examples/bench_compare \
  --output /tmp/hessboost-xgb-cuda --threads 16 --device cuda
```

See [Performance](../docs/performance.md#xgboost-comparison) for the recorded
comparison and workload definitions.
