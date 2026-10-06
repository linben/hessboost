# hessboost

[![crates.io](https://img.shields.io/crates/v/hessboost.svg)](https://crates.io/crates/hessboost)
[![docs.rs](https://img.shields.io/docsrs/hessboost)](https://docs.rs/hessboost)
[![CI](https://github.com/brndnmtthws/hessboost/actions/workflows/ci.yml/badge.svg)](https://github.com/brndnmtthws/hessboost/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

![hessboost](hessboost.png)

**Fast, deterministic gradient boosting in Rust (with Python bindings).**
hessboost provides multi-core tree building with runtime-detected NEON and AVX2
SIMD, strict parameter validation, reproducible models on any thread count, and
stable model storage. It supports modern extensions like conformal prediction,
explainable boosting machines (EBMs), distributional modeling, and tree-based
diffusion, alongside bidirectional XGBoost JSON/UBJSON model interchange.

The name comes from the Hessian: hessboost fits each tree to the loss's
gradients and second derivatives (Newton boosting).

## Why hessboost

- **Fast.** Multi-core training with runtime-detected NEON and AVX2 kernels;
  see [`docs/performance.md`](docs/performance.md).
- **Deterministic.** The same parameters, data, and seed produce the exact
  same model on any thread count.
- **Strict.** Invalid parameters and unsupported combinations fail loudly;
  nothing is silently ignored.
- **Stable storage.** Models saved natively are forwards-compatible across releases.
- **Modern modeling.** Built-in support for conformal intervals, Boulevard
  confidence bands, explainable boosting machines (EBMs), distributional
  boosting, SGLB uncertainty, tree-based diffusion, in-place updates, and
  compact models.
- **XGBoost compatible.** Accepts standard XGBoost parameter, objective,
  and metric names, with bidirectional JSON and UBJSON model interchange.

## Getting started

```sh
cargo add hessboost
```
For users pinning a release series in a Cargo manifest:

```toml
hessboost = "0.2"
```

Needs Rust 1.93 or newer and a C compiler (to build libzstd).

```rust
use hessboost::prelude::*;

fn main() -> Result<()> {
    // 100 rows × 4 features, row-major, and one label per row.
    let (n_rows, n_cols) = (100, 4);
    let x: Vec<f32> = (0..n_rows * n_cols).map(|i| (i % 17) as f32 / 17.0).collect();
    let y: Vec<f32> = x.chunks(n_cols).map(|row| 2.0 * row[0] - row[1]).collect();

    let dtrain = DMatrix::from_dense(&x, n_rows, n_cols)?.with_labels(&y)?;

    let params = TrainingParams::builder()
        .objective(Objective::SquaredError(RegLoss::default()))
        .tree_method(TreeMethod::Hist)
        .max_depth(6)
        .eta(0.1)
        .subsample(0.9)
        .build()?;

    let model = train(&params, &dtrain, 200)?;
    let preds = model.predict(&dtrain, Iterations::Best)?;
    println!("first prediction: {}", preds.get(0, 0).unwrap());

    model.save("model.bin", ModelFormat::Binary)?;
    let reloaded = BoostedModel::load("model.bin", ModelFormat::Binary)?;
    assert_eq!(reloaded.predict(&dtrain, Iterations::Best)?, preds);
    Ok(())
}
```

For eval sets, early stopping, custom objectives, or continued training, use
`Trainer` (the `xgb.train` keyword-argument equivalent):

```rust
use std::num::NonZeroUsize;

let result = Trainer::new(&params, &dtrain, 1000)
    .eval(&dvalid, "valid")
    .early_stopping_rounds(NonZeroUsize::new(20).unwrap())
    .train()?;
let model = result.model; // predicts with the best iteration
```

To ship a self-contained (e.g. static) binary, compile the model file into it;
it decodes on first use:

```rust
use hessboost::model::EmbeddedModel;

static MODEL: EmbeddedModel = EmbeddedModel::new(include_bytes!("model.bin"), ModelFormat::Binary);

let preds = MODEL.get()?.predict(&dtest, Iterations::Best)?;
```

Every type and option is in the [API docs](https://docs.rs/hessboost);
runnable programs live in [`examples/`](examples)
(`cargo run --release --example <name>`):

| Example | Shows |
|---|---|
| `train_regression` | end-to-end regression with feature importance |
| `binary_classification` | a watched eval set, early stopping, AUC |
| `balanced_bagging` | LightGBM class-stratified sampling for imbalanced binary classification |
| `multiclass` | per-class probabilities and predicted classes |
| `ranking` / `rank_xendcg` | LambdaMART and XE-NDCG with query bagging |
| `constraints` | monotone and interaction constraints, categorical features |
| `custom_objective` | a custom loss and eval metric |
| `shap` | SHAP contributions and interaction values |
| `model_io` | native and XGBoost JSON/UBJSON save and load |
| `conformal` | calibrated prediction intervals |
| `boulevard_inference` | confidence intervals for `f(x)` and prediction intervals |
| `ebm` | an explainable boosting machine's shape functions and their confidence bands |
| `distributional` | predictive distributions, intervals, and NLL |
| `virtual_ensembles` | SGLB posterior sampling: knowledge uncertainty rising off the training data |
| `tree_diffusion` | sampling multimodal and skewed `p(y \| x)` with tree diffusion and flow matching |
| `forest_flow` | synthetic tabular rows and imputation with ForestFlow / ForestDiffusion |
| `ordered_target_stats` | encoding a high-cardinality categorical |
| `compact_model` | reuse penalties and the compact model format |
| `budget` | budget training against default and tuned training |
| `online_update` | adding and deleting training rows in place, and exact unlearning |
| `pfn_boost` | boosting from a pretrained model's logits |
| `metal` | CPU vs GPU prediction (macOS, `--features metal`) |

## Python

[`python/`](python) holds the Python package (`pip install hessboost`):
`DMatrix`, `train`, `cv`, and `Booster` (taking XGBoost's parameter names),
scikit-learn estimators, pandas and polars categorical input, and the conformal,
distributional, tree-diffusion, ForestFlow, in-place update, ordered target
statistics, budget training, and compact model extras
(on macOS, `Booster.to_gpu()` batch-predicts on the Metal GPU):

```python
import hessboost

booster = hessboost.train(
    {"objective": "binary:logistic", "max_depth": 4}, hessboost.DMatrix(X, label=y), 100
)
probabilities = booster.predict(X_test)
```

See [`python/README.md`](python/README.md).

## Features

- **Core boosting:** `gbtree`, `dart`, and `gblinear` boosters, boosted random forests,
  and `exact`, `hist`, and `approx` tree methods with native missing-value and categorical support.
- **Objectives & metrics:** Regression (squared, log, Huber, quantile, expectile),
  binary/multiclass classification, ranking (LambdaMART, XE-NDCG), count, and survival (Cox, AFT),
  plus typed objective and metric APIs and custom loss hooks.
- **Validation & workflow:** Cross-validation (including purged and forward time-series folds,
  whole-query ranking folds, and per-fold target statistics), early stopping, feature importance,
  SHAP values and interactions, model slicing, and iteration ranges.
- **Interchange:** Native binary and JSON formats, XGBoost JSON/UBJSON import/export, LightGBM model import, and models embedded in the binary at compile time.
- **Modern modeling (opt-in):**
  - [Conformal intervals](https://docs.rs/hessboost/latest/hessboost/conformal/): Finite-sample coverage guarantees.
  - [Boulevard inference](https://docs.rs/hessboost/latest/hessboost/inference/): Asymptotic confidence and prediction intervals for `f(x)`.
  - [Explainable boosting machines](https://docs.rs/hessboost/latest/hessboost/ebm/): Interpretable cyclic GAM/GA²M models with shape-function confidence bands.
  - [Distributional boosting](https://docs.rs/hessboost/latest/hessboost/objective/distributional/): Full predictive distributions per row (`dist:normal`, `dist:gamma`, etc.).
  - [Uncertainty & virtual ensembles](https://docs.rs/hessboost/latest/hessboost/model/uncertainty/): SGLB posterior sampling and model shrinkage.
  - [Generative tabular modeling](https://docs.rs/hessboost/latest/hessboost/diffusion/): Tree-based conditional diffusion, flow matching, and ForestFlow synthetic data and imputation.
  - [In-place updates](https://docs.rs/hessboost/latest/hessboost/training/online/): Fast incremental learning and exact or approximate unlearning.
  - [Compact models](https://docs.rs/hessboost/latest/hessboost/model/compact/): Bit-packed model format with identical margins.
  - [Budget training](https://docs.rs/hessboost/latest/hessboost/training/budget/): Training controlled by one budget value, based on PerpetualBooster.
  - [Metal GPU](https://docs.rs/hessboost/latest/hessboost/backend/metal/): Apple Silicon GPU prediction and training (`--features metal`).
  - [CUDA GPU](https://docs.rs/hessboost/latest/hessboost/backend/cuda/): NVIDIA GPU training on Linux (`--features cuda`), bit-identical to the CPU; rows, histograms, and split scans stay on the GPU.

## Caveats

- Full technical details, invariants, and statistical assumptions are documented in the [API reference](https://docs.rs/hessboost).
- Approximate in-place updates are designed for incremental shifts (under ~1% of rows); larger changes benefit from a retrain.
- Asymptotic Boulevard inference and prediction intervals require specific noise and structure assumptions; see the [`inference` docs](https://docs.rs/hessboost/latest/hessboost/inference/#validation) for conditions and empirical coverage validation.
## Not implemented

- Distributed and external-memory training.
- CLI and C bindings.
- GPU training on Windows, and GPU prediction outside macOS.
- A few XGBoost options exist at one setting only, and a few metrics are
  missing; the [API docs](https://docs.rs/hessboost/latest/hessboost/#not-implemented)
  list them.

## Contributing

[`AGENTS.md`](AGENTS.md) has the build, lint, and test commands and the
project's invariants; [`scripts/README.md`](scripts/README.md) covers the
XGBoost parity suite and benchmark harnesses.

## License and attribution

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026
Brenden Matthews.

hessboost is a fork of
[sequoia-boost](https://github.com/pgarrett-scripps/sequoia-boost)
(Copyright 2026 Patrick Garrett, Apache-2.0).

hessboost is not affiliated with or endorsed by the
[XGBoost](https://github.com/dmlc/xgboost) project, and contains no XGBoost
source code.

Budget training reimplements
[PerpetualBooster](https://github.com/perpetual-ml/perpetual)'s algorithm
(Copyright 2024 Perpetual ML, Apache-2.0); no Perpetual code is copied.

The error function used by the AFT normal distribution is ported from
glibc 2.41's `s_erf.c`, derived from Sun Microsystems' fdlibm (Copyright (C)
1993 Sun Microsystems, Inc.); `src/objective/distributional/special.rs`
carries its notice.
