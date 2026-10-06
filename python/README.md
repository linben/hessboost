# hessboost for Python

**Fast, deterministic gradient boosting in Rust.** The core API (`DMatrix`,
`train`, `cv`, `Booster`, and scikit-learn estimators) takes the parameter
names XGBoost users know, and models move to and from XGBoost as JSON or
UBJSON.

- **Strict.** An unknown parameter, a value of the wrong type or range, or
  a combination hessboost does not implement raises an error; nothing is
  silently ignored.
- **Deterministic.** The same parameters, data, and seed give the same
  model at any `nthread`.
- **Typed** (`py.typed`, complete type information), with the GIL released
  while training and predicting, and free-threaded CPython supported.
- **Modern modeling (opt-in).** Conformal prediction intervals,
  confidence intervals for the regression function (Boulevard boosting),
  distributional boosting (a predictive distribution per row), LightGBM/CatBoost
  tree options, class-balanced binary bagging, and XE-NDCG ranking
  (`objective="rank:xendcg"`).

## Installation

```sh
uv add hessboost        # or: pip install hessboost
```

Extras: `hessboost[pandas]`, `hessboost[polars]`, and
`hessboost[scikit-learn]`. numpy is the only required dependency.

Prebuilt wheels are published for Linux x86_64 and aarch64 (glibc
manylinux and musl/Alpine musllinux), macOS arm64 (with Metal support:
`device="metal"` training histograms and `Booster.to_gpu()` GPU batch
prediction), and Windows x86_64: one `abi3` wheel per platform for
CPython 3.11 and newer, plus a wheel for free-threaded CPython 3.14t.
Elsewhere the installer builds from the source distribution, which needs
Rust 1.93 or newer and a C compiler (for libzstd).

## Quick start

Train on a `DMatrix`, evaluating a holdout set every round and stopping
once it stops improving:

```python
import numpy as np
import hessboost

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 5))
y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(int)

dtrain = hessboost.DMatrix(X[:800], label=y[:800])
dvalid = hessboost.DMatrix(X[800:], label=y[800:])

booster = hessboost.train(
    {"objective": "binary:logistic", "max_depth": 4, "eta": 0.1, "eval_metric": "auc"},
    dtrain,
    num_boost_round=500,
    evals=[(dvalid, "valid")],
    early_stopping_rounds=20,
    verbose_eval=False,
)
```

`predict` takes a `DMatrix` or anything its constructor accepts. The default
range is the iterations through `best_iteration` (pass `iteration_range=(0, 0)`
for every iteration), except `pred_leaf`, which defaults to all of the
model's trees, not just through `best_iteration`.
The flag arguments are exclusive:

```python
probabilities = booster.predict(X[800:])  # through best_iteration
margins = booster.predict(X[800:], output_margin=True)
shap = booster.predict(X[800:], pred_contribs=True)  # (rows, features + 1)
leaves = booster.predict(X[800:], pred_leaf=True)  # (rows, trees) int32, all trees
print(booster.best_iteration, booster.get_score(importance_type="gain"))
```

### Input data

`DMatrix` takes numpy arrays of any numeric dtype and memory layout (a
C-contiguous `float32` array is used without a copy), pandas and polars
DataFrames, scipy sparse matrices, and anything `numpy.asarray` accepts.
NaN, and a frame's null, is missing (or pass `missing=`); ranking data
takes `group=` sizes or `qid=`, and `survival:aft` takes
`label_lower_bound=`/`label_upper_bound=`.
`Booster.predict` accepts the same inputs directly.

### Training controls

`train` supports XGBoost's everyday arguments: `evals`, `evals_result`,
`early_stopping_rounds`, `verbose_eval` (printed live, every round or
every `n`-th), `xgb_model` (continued training, or tree refresh with
`process_type="update"`), a custom objective `obj`, a `custom_metric`,
and `callbacks`. Ctrl-C stops training at the end of the current round
and raises `KeyboardInterrupt`. `hessboost.cv` cross-validates over
shuffled folds, explicit folds, or a scikit-learn splitter; on ranking data
each fold must hold whole query groups (e.g. `GroupKFold` over the query
ids). `hessboost.train_with_budget(params, dtrain, budget)` trains with one
fitting budget in place of `eta`, tree limits, and a round count
(PerpetualBooster's algorithm).

Subclass `TrainingCallback` to stop on your own condition; `after_iteration`
sees the round and the evaluation history, and returning `True` stops
training with the rounds so far:

```python
class StopAtTarget(hessboost.TrainingCallback):
    def after_iteration(self, iteration, evals_log):
        return evals_log["valid"]["auc"][-1] > 0.99  # True stops training


hessboost.train(
    {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 4},
    dtrain,
    1000,
    evals=[(dvalid, "valid")],
    callbacks=[StopAtTarget()],
)
```

### DataFrames and categorical features

pandas and polars column names become feature names, and categorical
columns (pandas `category`; polars `Enum` and `Categorical`) become native
categorical features. A category's code is its position in the column's
categories: a pandas column's or an `Enum`'s declared categories, or a
polars `Categorical`'s values, sorted (as pandas infers them). The booster
remembers each column's categories and re-codes a frame passed to `predict`
(and the other prediction and calibration methods) whose categories are
ordered differently or include unseen values (which count as missing),
whichever library the frame is from:

```python
import pandas as pd

df = pd.DataFrame({"color": pd.Categorical(["red", "blue", "red"] * 100), "size": np.arange(300.0)})
booster = hessboost.train({}, hessboost.DMatrix(df, label=np.arange(300.0)), 20)
booster.predict(df)
```

A `DMatrix` is coded once, when it is built, so it must have the features
of every model or matrix it meets: `predict` and the calibrators check it
against each model reading it, and `train` checks every `evals` matrix
against `dtrain` and against `xgb_model`, and `dtrain` against `xgb_model`
(continued training or refresh). Feature names (where both have them;
`predict(validate_features=False)` skips them), which features are
categorical, and each categorical feature's categories, in order, must
match; a mismatch raises `HessboostError` naming the eval set and the
feature. Build eval frames with the training frame's categories (for
example `valid["color"].cat.set_categories(train["color"].cat.categories)`
in pandas; in polars, an `Enum` of the training categories, since a
`Categorical` missing some of its values has other categories).
Codes without recorded categories (numpy data with
`feature_types=["c", ...]`) are taken to be the other side's codes; only
which features are categorical is compared, and a model continued on them
keeps the earlier model's categories. `ConformalizedQuantile.calibrate`
likewise needs its two models to share their features, and re-codes frames
to whichever model records categories. The scikit-learn estimators re-code
every `eval_set` frame to the training frame's categories and, with
`xgb_model`, the training frame to the earlier model's.

## scikit-learn

The estimators take XGBoost's scikit-learn parameter names, plus
`callbacks`. Parameters left at `None` keep hessboost's defaults, and
`params={...}` passes any other training parameter; `fit(..., verbose=True)`
(or a period) prints rounds live:

```python
from hessboost.sklearn import HessboostClassifier

model = HessboostClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.1, early_stopping_rounds=20
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
model.predict_proba(X_test)
model.feature_importances_
```

`HessboostRegressor` (multi-target with a 2-D `y`), `HessboostClassifier`
(any class labels), `HessboostRanker` (`group=` or `qid=`), and
`HessboostDistributionRegressor` (`dist:*` objectives; `predict` returns
means, `predict_distribution` the distributions) work with pipelines,
`clone`, grid search, and pickling, and pass scikit-learn's estimator checks
(except three documented deviations). `import hessboost` does not import
scikit-learn; `hessboost.sklearn` needs it.

## Model files

| Method | Formats |
|---|---|
| `save_model(path, format=None)` | by extension: `.json` native JSON, `.ubj` XGBoost UBJSON, anything else native binary; or `format="binary" \| "json" \| "xgboost-json" \| "xgboost-ubjson"` |
| `Booster(path_or_bytes)`, `load_model(...)` | any of the four, or a LightGBM 4.x text model (`format="lightgbm"`, import only), detected from the content (`ModelFormatError` if it looks like none of them) |
| `save_raw(format="binary")` | the same formats as bytes |
| `pickle` / `copy` | native binary plus feature names, categories, and `best_score` |

The native binary format is compressed, checksummed, and lossless; files
load in subsequent releases. A LightGBM model
(`lightgbm.Booster.save_model`) predicts LightGBM's values for inputs with
missing values as `NaN` and categorical features as non-negative codes;
models with no exact equivalent raise `ModelFormatError`:

```python
booster.save_model("model.ubj")  # a file XGBoost loads; .json for native JSON
loaded = hessboost.Booster("model.ubj")  # format detected from the content
early = booster[:10]  # the first 10 boosting iterations, as a Booster
```

## Modern modeling

### Conformal intervals

`SplitConformal` wraps one single-output model in `f(x) ± Q`, where `Q` is
the calibration set's quantile of `|y - f(x)|` — finite-sample `1 - alpha`
coverage for exchangeable rows:

```python
from hessboost.conformal import SplitConformal

cal = SplitConformal.calibrate(booster, X_cal, y_cal, alpha=0.1)
lower, upper = cal.predict_interval(X_test).T
```

`ConformalizedQuantile` instead adjusts a quantile band `[q_lo(x), q_hi(x)]`
by its calibration quantile, so it also tightens a band that over-covers.
The band comes from two single-quantile models (`calibrate`), two outputs
of one multi-quantile model (`calibrate_outputs`), or a `dist:*` model's
central quantiles (`calibrate_distribution`):

```python
from hessboost.conformal import ConformalizedQuantile

band = hessboost.train(
    {"objective": "reg:quantileerror", "quantile_alpha": [0.05, 0.95]},
    hessboost.DMatrix(X_train, y_train),
    200,
)
cqr = ConformalizedQuantile.calibrate_outputs(band, X_cal, y_cal, alpha=0.1)
lower, upper = cqr.predict_interval(X_test).T  # >= 90% coverage, finite-sample
```

### Distributional boosting

A `dist:*` objective fits a predictive distribution per row instead of a
point. `predict_distribution` returns a `Distributions` object whose
summaries are vectorized over rows:

```python
dist = hessboost.train({"objective": "dist:normal"}, hessboost.DMatrix(X_train, y_train), 300)
d = dist.predict_distribution(X_test)
means, stds = d.mean(), d.std()
lower, upper = d.interval(0.9).T
ll, score = d.log_prob(y_test), d.crps(y_test)
```

### Virtual ensembles

A model trained with `posterior_sampling` (SGLB: `langevin` noise plus
`model_shrink_rate`) carries its own posterior ensemble: `count` members
rebuilt exactly from its model shrinkage, member-major (`(count, rows)` for
single-output models), with each member's iteration count alongside:

```python
sglb = hessboost.train({"posterior_sampling": True}, hessboost.DMatrix(X_train, y_train), 1000)
members, iterations = sglb.predict_virtual_ensembles(X_test, 10)
u = sglb.predict_uncertainty(X_test, 10)  # u.mean, u.knowledge, u.data, u.total
```

`predict_uncertainty` decomposes the ensemble à la CatBoost: knowledge
(epistemic) uncertainty rises off the training data. Plain regression has
only `mean` and `knowledge` (`data`/`total` are `None`); `dist:*` and
classification models get the full decomposition.

### Online updates

`OnlineModel` keeps a trained model with its training data and updates both
in place as rows arrive or must be forgotten. `Exact()` retrains, so every
update equals `hessboost.train` on the updated data bit for bit;
`Approximate(tolerance)` (default, `0.1`) keeps splits that still rank near
the top and is faster for small changes. A refused, stopped, or interrupted
(Ctrl-C) update changes nothing:

```python
from hessboost.online import Approximate, OnlineModel

online = OnlineModel.train(
    {"tree_method": "hist", "max_depth": 6},
    hessboost.DMatrix(X_train, y_train),
    100,
    Approximate(0.1),
)
report = online.update(hessboost.DMatrix(X_new, y_new), deletions=[3, 17])
online.model.predict(X_test)  # online.data: the updated training rows
```

The report counts kept nodes, regrown subtrees, and refreshed rows
(`UpdateReport`); `OnlineModel.from_model` resumes from a saved `Booster`
and its training data. Updates need `hist` depth-wise trees without sampling
or constraints and unweighted data.

### Explainable boosting machines

`{"booster": "ebm"}` trains a cyclic GA2M with outer bags, FAST pairs, and
per-bag early stopping. Every term's piecewise-constant shape function is
readable from the model (`NumericAxis` edges or `CategoricalAxis` codes,
plus a missing cell per axis); intercept plus shapes is the margin:

```python
from hessboost import ebm

model = hessboost.train({"booster": "ebm"}, hessboost.DMatrix(X_train, y_train), 50)
shapes = ebm.shape_functions(model)
shapes.intercept, shapes.terms[0].values
```

A Boulevard EBM (`{"booster": "ebm", "ebm_boulevard": True}`) additionally
supports confidence bands on its shapes via
`hessboost.inference.EbmInference.term_bands`.

### Boulevard inference

Boulevard boosting (`{"booster": "boulevard"}`) samples trees with dropout
so the ensemble converges to a kernel ridge posterior, whose leaf kernel
gives asymptotic confidence, prediction, and reproduction intervals for
`f(x)`. Refit the leaves on independent rows first (`honest_refit`), fit
the kernel over those rows, then query any rows:

```python
from hessboost.inference import BoulevardInference, honest_refit

trained = hessboost.train(
    {"booster": "boulevard", "eta": 0.8, "boulevard_dropout": 0.5, "subsample": 0.8},
    hessboost.DMatrix(X_struct, y_struct),
    200,
)
model = honest_refit(trained, X_values, y_values)  # leaves from independent rows
inference = BoulevardInference.fit(model, X_values, y_values, holdout=X_cal, holdout_label=y_cal)
lower, upper = inference.confidence_intervals(X_test, alpha=0.05).T  # for f(x)
```

The intervals are conditional on the tree structures: nominal for
low-dimensional smooth signals after an honest refit, under-covering
elsewhere (see the crate's `inference` docs). `Booster.boulevard` records
how the model was trained.

### Diffusion models

`hessboost.diffusion` fits nonparametric `p(y | x)` for scalar or vector
labels (multimodal, skewed, heavy-tailed) by conditional diffusion or flow
matching with GBDT score models, after Treeffuser and DiffGBM. `sample`
returns `(rows, n_samples, outputs)` draws, deterministic per seed and step
count (`n_steps=` overrides the model's); `mean`, `quantiles`, and `crps`
summarize them:

```python
from hessboost.diffusion import DiffusionModel, DiffusionParams, crps, quantiles

flow = DiffusionModel.fit(DiffusionParams.flow_matching(), X_train, y_train)
draws = flow.sample(X_test, 200, seed=0)  # (rows, 200, outputs) float32
bands = quantiles(draws, [0.05, 0.5, 0.95])  # (rows, 3, outputs)
scores = crps(draws, y_test)  # (rows, outputs)
```

`DiffusionParams` is a frozen dataclass tree with presets `default()`,
`treeffuser()`, and `flow_matching()`; its `training` mappings are XGBoost
parameters, as `train` reads them. Models save with
`to_bytes(format="binary")` / `save(path, format="binary")` (`"binary"` or
`"json"`) and load with `from_bytes(data, format="auto")` /
`load(path, format="auto")`, or pickle (which also keeps feature names and
categories). `fit` releases the GIL but cannot be interrupted: Ctrl-C takes
effect once it returns.

### Synthetic tabular data

`hessboost.diffusion.forest` fits ForestFlow / ForestDiffusion models of
whole rows (optionally per class of a label) with per-noise-level GBDTs.
`sample` draws synthetic rows as a `ForestSamples` (`values`, plus `labels`
for a class-conditional model); `sample_for_labels` draws one row per given
label:

```python
from hessboost.diffusion.forest import ForestModel, ForestParams

forest = ForestModel.fit(ForestParams.forest_diffusion(), X_with_nans)
synthetic = forest.sample(1000, seed=0)  # .values (1000, columns), .labels None
```

The diffusion variant also imputes missing values, keeping the observed
entries (flow models refuse). Values are in the data's own coding:

```python
filled = forest.impute(X_with_nans, n_imputations=5)  # (5, rows, columns)
```

`ForestParams` has presets `forest_flow()` (the defaults) and
`forest_diffusion()`; its `training` mappings are XGBoost parameters, and
models save and load like `DiffusionModel`s.

### GPU prediction

On macOS, `Booster.to_gpu()` lays the model out for batch prediction on
Metal: the forest uploads once, and each call predicts bit-identically to
`Booster.predict` (values or raw margins), faster from roughly a few
thousand row-trees upward:

```python
from hessboost import GpuModel

gpu = booster.to_gpu()
probabilities = gpu.predict(X_test)
```

### Compact models

`Booster.to_compact()` packs the trees default prediction uses into
hessboost's bit-packed `HBTD` format (*Boosted Trees on a Diet*), which
predicts bit-identical values and margins in a fraction of the size;
training with `toad_penalty_feature`/`toad_penalty_threshold` shrinks it
further. `Booster.size_report()` compares the two formats:

```python
from hessboost import CompactModel

compact = booster.to_compact()
compact.save_model("model.hbtd")
predictions = CompactModel("model.hbtd").predict(X_test)
print(booster.size_report().compression_ratio)
```

### Validation folds

`hessboost.folds` builds `(train_rows, test_rows)` splits for `cv` or custom
loops: shuffled `k_fold` (what `cv` uses by default), `forward_chaining`
expanding windows with a purged `gap`, and `purged_forward` (timestamped
rows, purged by each row's own label window, for overlapping or irregular
horizons):

```python
from hessboost import folds

splits = folds.forward_chaining(dtrain.num_row(), 4, gap=24)
result = hessboost.cv({"max_depth": 4}, dtrain, 100, folds=splits)
```

### Ordered target statistics

`hessboost.target_stats.OrderedTargetEncoder` replaces categorical columns
with CatBoost-style ordered target means: a training row's encoding never
sees its own label. `label=` supplies the target for a multi-target matrix
or a class's 0/1 indicator. In `cv`, `target_stats=` fits the encoder on
each fold's training rows only, so no held-out label reaches an encoding:

```python
from hessboost.target_stats import OrderedTargetEncoder

encoder = OrderedTargetEncoder(seed=7)
dtrain_encoded, stats = encoder.fit_transform(dtrain, ["city"])
booster = hessboost.train({"max_depth": 4}, dtrain_encoded, 100)
predictions = booster.predict(stats.transform(X_test))
result = hessboost.cv({"max_depth": 4}, dtrain, 100, target_stats=["city"], target_encoder=encoder)
```

### Extra training options

Every hessboost training option (`path_smooth`, `extra_trees`,
`linear_tree`, `grow_policy="symmetric"`, `use_quantized_grad`,
`pos_bagging_fraction`, `neg_bagging_fraction`, `bagging_by_query`, the
`dist:*` objectives and their `dist_gradient`, `rank:xendcg`, ...) is a
`params` key. XE-NDCG's keyed per-round random stream differs from
LightGBM's `rank_xendcg` stream.

## Differences from XGBoost's Python package

- Errors: `hessboost.HessboostError` (a `ValueError`) for refused inputs,
  with subclasses `InvalidDataError` (data content: labels outside the
  objective's domain, negative weights, bad groups or bounds; the message
  names the input and any eval set), `IncompatibleModelError` (a model the
  parameters or data do not match: continued training, refresh, slicing)
  and `ModelFormatError` (model files); invalid or conflicting parameters
  raise `HessboostError` itself. `TypeError` for wrong types.
  Unsupported parameters are refused, including `verbosity`; `missing`
  belongs to `DMatrix`.
- `TrainingCallback.after_iteration(iteration, evals_log) -> bool` sees the
  round and the evaluation history, not the model (XGBoost's also gets the
  booster and has `before_training`/`after_training` hooks); returning
  `True` stops training with the rounds so far. `hessboost.cv` has no
  callbacks and finishes its folds before Ctrl-C takes effect.
- `custom_metric(predictions, labels, weights)` returns a float and is named
  by the function's `__name__` (XGBoost passes a `DMatrix` and returns
  `(name, value)`). `obj(margins, dtrain)` matches XGBoost; the model then
  predicts margins from a zero intercept (or `base_score`). `obj` replaces
  `objective`, which `params` must then not set, and `num_class` is the
  custom objective's output count (XGBoost's custom-softmax convention;
  default: one per label column).
  Unlike XGBoost, which accepts both, hessboost refuses `objective` alongside
  `obj`: drop `objective` from ported `params`; the callback needs no change.
- `cv` returns a dict of numpy arrays (`test-<metric>-mean`/`-std`) with
  held-out metrics only; there is no `stratified` or `as_pandas`.
- `predict` defaults to the iterations through `best_iteration` (XGBoost's
  scikit-learn behavior); pass `iteration_range=(0, 0)` for all. SHAP and
  leaf ranges start at iteration 0; `pred_leaf` defaults to every iteration
  instead, and returns `int32`.
- On macOS, `Booster.to_gpu()` lays the model out for GPU batch prediction
  on Metal (`GpuModel.predict`, bit-identical to `Booster.predict`; see
  whether a device exists with `GpuModel.available()`).
- Model files do not store feature names or categories (pickles do).
- Not available: `DMatrix` from files or `QuantileDMatrix`, `inplace_predict`
  (`predict` takes arrays directly), `Booster.get_dump`/`trees_to_dataframe`
  /`dump_model`, attributes (`set_attr`), plotting, distributed (Dask/Spark)
  and CUDA training (the Rust crate's `cuda` feature is not in the wheels
  yet), `approx_contribs`, and `strict_shape`. The macOS wheels support
  `device="metal"` (GPU histograms while training).

## Development

From `python/` in the [repository](https://github.com/brndnmtthws/hessboost),
with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked              # build the extension and install dev tools
uv run --locked pytest
uv run --locked pyright --verifytypes hessboost --ignoreexternal
```

Lint, format, and type-check all of the repository's Python from its root
(configuration: `ruff.toml`, `ty.toml`):

```sh
uv run --project python --locked ruff check
uv run --project python --locked ruff format --check
uv run --project python --locked ty check
```

## License

Apache-2.0.
