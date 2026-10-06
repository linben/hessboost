//! Training configuration.
//!
//! Parameter names and default values deliberately mirror XGBoost so that
//! existing knowledge and configurations transfer directly. Where XGBoost
//! exposes aliases (e.g. `eta`/`learning_rate`), we pick the canonical field
//! name and document the alias.

use super::groups::{
    BalancedBagging, Boulevard, Dart, Ebm, ExtraTrees, Langevin, LinearTree, ModelShrink,
    ModelShrinkMode, QuantizedGrad, QueryBagging, Refresh,
};
use crate::check::{ensure, narrows, non_negative, positive, unit};
use crate::error::{HessboostError, Result};
use crate::objective::{Loss, LossContext, Objective};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;

/// The bound on each leaf weight's absolute value. XGBoost
/// `max_delta_step`.
///
/// XGBoost reads an unset `max_delta_step` as the objective's default and
/// an explicit `0` as "no bound", so the three states stay distinct: for
/// [`Objective::Poisson`] the default is `0.7` (the same value also
/// stabilizes the Poisson Hessian), and [`Unbounded`](Self::Unbounded)
/// turns that off.
///
/// ```
/// use hessboost::config::MaxDeltaStep;
/// use hessboost::prelude::*;
///
/// # fn main() -> hessboost::error::Result<()> {
/// let bounded = TrainingParams::builder()
///     .max_delta_step(MaxDeltaStep::Bounded(0.5))
///     .build()?;
/// assert_eq!(bounded.max_delta_step, MaxDeltaStep::Bounded(0.5));
/// // A bound must be positive: no bound is `Unbounded`.
/// assert!(
///     TrainingParams::builder()
///         .max_delta_step(MaxDeltaStep::Bounded(0.0))
///         .build()
///         .is_err()
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum MaxDeltaStep {
    /// The objective's default: `0.7` for [`Objective::Poisson`], otherwise
    /// no bound (also for a custom loss). XGBoost's unset `max_delta_step`.
    #[default]
    ObjectiveDefault,
    /// No bound, even where the objective has a default. XGBoost
    /// `max_delta_step = 0`.
    Unbounded,
    /// Every leaf weight lies in `[-v, v]`; `v` must be positive and finite
    /// once rounded to `f32`.
    Bounded(f64),
}

impl MaxDeltaStep {
    /// The bound in effect for `objective`, with XGBoost's `0` for none.
    pub(crate) fn resolve(self, objective: &Objective) -> f64 {
        match self {
            MaxDeltaStep::ObjectiveDefault => objective.default_max_delta_step(),
            MaxDeltaStep::Unbounded => 0.0,
            MaxDeltaStep::Bounded(v) => v,
        }
    }
}

/// Which booster to use in the ensemble.
///
/// Mirrors XGBoost's `booster` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum BoosterKind {
    /// Gradient boosted trees (XGBoost `gbtree`).
    #[default]
    GbTree,
    /// Dropout Additive Regression Trees (XGBoost `dart`) with this
    /// dropout.
    Dart(Dart),
    /// Linear booster with coordinate descent (XGBoost `gblinear`). It
    /// updates from every row and feature and grows no trees, so row and
    /// column sampling, `num_parallel_tree > 1`, tree constraints, and
    /// training-matrix feature weights are refused with it.
    GbLinear,
    /// Boulevard boosting for statistical inference (opt-in):
    /// every iteration's trees are averaged rather than summed, so the
    /// ensemble converges to a kernel ridge regression with a central limit
    /// theorem. `num_parallel_tree = 1` runs BRAT-D (Fang, Tan & Hooker,
    /// NeurIPS 2025, Algorithm 1; Zhou & Hooker's Boulevard at
    /// [`Boulevard::dropout`] `= 0`), more
    /// trees per iteration BRAT-P (Algorithm 2). Squared-error regression
    /// only; see [`crate::inference`] for the trained model's confidence and
    /// prediction intervals and the settings it refuses.
    Boulevard(Boulevard),
    /// Explainable boosting machine (EBM, a GA²M; opt-in):
    /// cyclic boosting of one small tree per feature at a time, so the
    /// model is a sum of per-feature shape functions, optionally followed
    /// by pairwise interaction terms (FAST detection,
    /// [`Ebm::interactions`]) and outer
    /// bagging. With [`Ebm::boulevard`] the
    /// terms are Boulevard-averaged instead, which gives the shape
    /// functions confidence bands. See [`crate::ebm`] for the algorithms,
    /// the shape functions, and the settings it refuses.
    Ebm(Ebm),
}

/// Tree construction algorithm.
///
/// Mirrors XGBoost's `tree_method`. `Auto` resolves to [`TreeMethod::Hist`] for
/// all but the smallest datasets, matching modern XGBoost behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum TreeMethod {
    /// Pick automatically based on dataset size.
    #[default]
    Auto,
    /// Exact greedy algorithm (enumerate every split candidate).
    Exact,
    /// Approximate algorithm using weighted quantile sketch per split.
    Approx,
    /// Fast histogram algorithm with pre-binned features.
    Hist,
}

/// Order in which the tree is grown.
///
/// Mirrors XGBoost's `grow_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum GrowPolicy {
    /// Split nodes closest to the root first (level-wise). XGBoost default.
    #[default]
    DepthWise,
    /// Split nodes with the highest loss reduction first (leaf-wise).
    LossGuide,
    /// Symmetric (oblivious) trees, CatBoost-style: every level applies one
    /// shared split (feature, threshold, missing direction) chosen to maximize
    /// the summed gain over the level's nodes. Opt-in. Needs
    /// a tree booster (`gbtree` or `dart`), `tree_method = hist` or `approx`,
    /// numerical features only, `max_depth`
    /// in `1..=`[`MAX_SYMMETRIC_DEPTH`], and no `max_leaves`. A node whose
    /// level split would violate `min_child_weight`, `gamma`, or a monotone
    /// constraint stays a leaf. The trees are ordinary [`RegTree`]s, so they
    /// export to XGBoost unchanged; prediction routes rows through them by
    /// bit pattern.
    ///
    /// [`RegTree`]: crate::tree::RegTree
    Symmetric,
}

/// Which processor training runs on. XGBoost `device`: `cpu`, `cuda`,
/// `cuda:<ordinal>`, and XGBoost's aliases `gpu`/`gpu:<ordinal>` for CUDA;
/// the macOS GPU backend is `metal`. Serializes as those strings (`cuda`
/// for ordinal 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Device {
    /// The CPU (default): always available, and what the parity fixtures
    /// run on.
    #[default]
    Cpu,
    /// Apple's Metal GPU, on macOS 10.15 or later with the `metal` feature:
    /// histogram construction runs on the GPU for every node whose sums it
    /// can compute exactly and on the CPU for the rest, reproducing
    /// single-threaded CPU training bit for bit. Requires `tree_method =
    /// hist`/`auto` and a tree booster. Opt-in.
    ///
    /// A correctness path so far, not a speedup: with the earlier
    /// floating-point kernels the GPU histograms were slower than the
    /// multicore CPU's, and the current integer kernels are unmeasured
    /// (see [`backend::metal`](crate::backend::metal)); the fast Metal path is
    /// prediction, through
    /// [`BoostedModel::to_gpu`](crate::model::BoostedModel::to_gpu).
    Metal,
    /// An NVIDIA GPU through CUDA, on Linux with the `cuda` feature (see
    /// [`backend::cuda`](crate::backend::cuda)): the tree's rows,
    /// partitions, and histograms live on the GPU (and, for squared error
    /// and logistic objectives in a plain `gbtree`, the margins and
    /// gradients), reproducing single-threaded CPU training bit for bit.
    /// Requires `tree_method = hist`/`auto` and a tree booster. Opt-in.
    Cuda {
        /// The CUDA device ordinal (XGBoost `cuda:<ordinal>`; `0` for
        /// plain `cuda`).
        ordinal: usize,
    },
}

impl Device {
    /// The XGBoost spellings [`Device`] parses, for error messages.
    const SPELLINGS: &'static [&'static str] = &[
        "cpu",
        "metal",
        "cuda",
        "cuda:<ordinal>",
        "gpu",
        "gpu:<ordinal>",
    ];

    /// Parse an XGBoost `device` value.
    fn parse(value: &str) -> Option<Self> {
        match value {
            "cpu" => return Some(Device::Cpu),
            "metal" => return Some(Device::Metal),
            "cuda" | "gpu" => return Some(Device::Cuda { ordinal: 0 }),
            _ => {}
        }
        let ordinal = value
            .strip_prefix("cuda:")
            .or_else(|| value.strip_prefix("gpu:"))?;
        // Digits only: `usize::from_str` would also take a leading `+`.
        if ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        ordinal.parse().ok().map(|ordinal| Device::Cuda { ordinal })
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => f.write_str("cpu"),
            Device::Metal => f.write_str("metal"),
            Device::Cuda { ordinal: 0 } => f.write_str("cuda"),
            Device::Cuda { ordinal } => write!(f, "cuda:{ordinal}"),
        }
    }
}

impl Serialize for Device {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Device {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Device::parse(&value)
            .ok_or_else(|| serde::de::Error::unknown_variant(&value, Device::SPELLINGS))
    }
}

/// Deepest tree `grow_policy = symmetric` grows (`2^16` leaves), CatBoost's
/// depth limit.
pub const MAX_SYMMETRIC_DEPTH: usize = 16;

/// Largest [`TrainingParams::num_parallel_tree`]: an iteration's forest is
/// grown and held in memory at once (with one row sample per tree), so the
/// count is bounded well below what its bookkeeping could address.
pub(crate) const MAX_NUM_PARALLEL_TREE: usize = 1 << 16;

/// Per-feature monotonicity direction: XGBoost's `-1`/`0`/`1`, a complete
/// set, so it can be matched exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Monotone {
    /// No constraint on this feature.
    #[default]
    None,
    /// Prediction must be non-decreasing in this feature.
    Increasing,
    /// Prediction must be non-increasing in this feature.
    Decreasing,
}

/// How rows are subsampled each round.
///
/// Mirrors XGBoost's `sampling_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SamplingMethod {
    /// Every row is kept with probability `subsample`. XGBoost default.
    #[default]
    Uniform,
    /// Minimal-variance sampling (XGBoost `gradient_based`): each tree keeps
    /// row `i` with probability `min(1, sqrt(g_i^2 + 0.1 h_i^2) / u)`, where
    /// `u` makes the expected kept count `trunc(n * subsample)`, and scales a
    /// kept row's gradient and Hessian by the inverse of that probability.
    /// Supported by `tree_method = hist | approx | auto`; `exact` rejects it
    /// when `subsample < 1`.
    GradientBased,
}

/// How multi-target and multiclass models allocate outputs to trees.
///
/// Mirrors XGBoost's `multi_strategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MultiStrategy {
    /// One tree per output each round. XGBoost default.
    #[default]
    OneOutputPerTree,
    /// One tree per round whose leaves hold a vector of all outputs
    /// (vector-leaf trees; `tree_method = hist` only). With a single output
    /// it trains scalar trees, like XGBoost.
    ///
    /// Vector-leaf trees are refused by the options that replace or bypass
    /// the XGBoost split search: symmetric growth, the reuse penalties,
    /// quantized gradients, `extra_trees`, `path_smooth`, `linear_tree`,
    /// budget mode ([`training::budget`](crate::training::budget)), and a
    /// GPU [`device`](TrainingParams::device). The refresh updater
    /// (`process_type = update`) and the compact format
    /// ([`model::compact`](crate::model::compact)) refuse vector-leaf
    /// models. Reduced split gradients
    /// ([`Loss::split_gradient`](crate::objective::Loss::split_gradient))
    /// cannot be combined with monotone constraints.
    MultiOutputTree,
}

/// Whether a round grows new trees or updates existing ones.
///
/// Mirrors XGBoost's `process_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ProcessType {
    /// Grow new trees. XGBoost default.
    #[default]
    Default,
    /// Revisit the trees of an existing model instead of growing new ones
    /// (the refresh updater; see
    /// [`Trainer::init_model`](crate::training::Trainer::init_model)). It
    /// keeps every split and sums every row, so settings it does not read
    /// (row and column sampling, symmetric growth, DART dropout, the
    /// LightGBM and compact-training tree options) and training-matrix feature weights
    /// must keep their defaults.
    Update(Refresh),
}

/// The complete training configuration.
///
/// Construct with [`TrainingParams::builder`], start from
/// [`TrainingParams::default`] and mutate fields directly, or parse
/// XGBoost's flat key/value form with [`TrainingParams::from_xgboost`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TrainingParams {
    // ---- General ----
    /// Which booster to train. XGBoost `booster`.
    pub booster: BoosterKind,
    /// Number of worker threads; `None` uses the global Rayon pool.
    /// XGBoost `nthread` (`0` there is `None` here).
    pub nthread: Option<NonZeroUsize>,
    /// RNG seed for subsampling and column sampling. XGBoost `seed`.
    pub seed: u64,

    /// Which processor training runs on. XGBoost `device`. `metal` (macOS,
    /// `metal` feature) and `cuda` (Linux, `cuda` feature) move histogram
    /// construction to the GPU; the default `cpu` leaves everything as it
    /// was.
    pub device: Device,

    // ---- Learning task ----
    /// The learning objective with its parameters, or a custom loss
    /// ([`Objective::Custom`]). XGBoost `objective` (plus the parameters
    /// each objective reads: `num_class`, `scale_pos_weight`,
    /// `tweedie_variance_power`, ...).
    pub objective: Objective,
    /// Global bias / initial prediction (in probability space where applicable).
    /// `None` means "estimate from the labels", matching modern XGBoost.
    /// XGBoost `base_score`.
    pub base_score: Option<f64>,
    /// The metrics evaluated on every eval set, in order (the last one
    /// drives early stopping). Empty means the loss's default
    /// ([`Loss::default_metric`](crate::objective::Loss::default_metric)).
    /// XGBoost `eval_metric`.
    pub eval_metric: Vec<crate::metric::EvalMetric>,

    // ---- Tree booster ----
    /// Learning rate / step-size shrinkage. XGBoost `eta` / `learning_rate`.
    pub eta: f64,
    /// Minimum loss reduction to make a split. XGBoost `gamma` / `min_split_loss`.
    pub gamma: f64,
    /// Maximum tree depth; `None` is no limit. XGBoost `max_depth` (`0`
    /// there is `None` here).
    pub max_depth: Option<NonZeroUsize>,
    /// Maximum number of leaves per tree grown by `lossguide` (and of
    /// vector-leaf trees under either policy); `None` is no limit. As in
    /// XGBoost, depth-wise scalar trees read only `max_depth`. XGBoost
    /// `max_leaves` (`0` there is `None` here).
    pub max_leaves: Option<NonZeroUsize>,
    /// Minimum sum of instance hessian needed in a child. XGBoost `min_child_weight`.
    pub min_child_weight: f64,
    /// The bound on each leaf weight ([`MaxDeltaStep`]). XGBoost
    /// `max_delta_step`.
    pub max_delta_step: MaxDeltaStep,
    /// Row subsample ratio per boosting round. XGBoost `subsample`.
    pub subsample: f64,
    /// Column subsample ratio per tree. XGBoost `colsample_bytree`.
    pub colsample_bytree: f64,
    /// Column subsample ratio per level. XGBoost `colsample_bylevel`.
    pub colsample_bylevel: f64,
    /// Column subsample ratio per node. XGBoost `colsample_bynode`.
    pub colsample_bynode: f64,
    /// L2 regularization on leaf weights. XGBoost `lambda` / `reg_lambda`.
    pub lambda: f64,
    /// L1 regularization on leaf weights. XGBoost `alpha` / `reg_alpha`.
    pub alpha: f64,
    /// Tree construction algorithm. XGBoost `tree_method`.
    pub tree_method: TreeMethod,
    /// Tree growth order. XGBoost `grow_policy`.
    pub grow_policy: GrowPolicy,
    /// Maximum number of histogram bins per feature. XGBoost `max_bin`.
    pub max_bin: usize,
    /// Per-feature monotone constraints (empty = none). XGBoost `monotone_constraints`.
    pub monotone_constraints: Vec<Monotone>,
    /// Allowed feature-interaction groups (empty = none). Each inner vector lists
    /// feature indices permitted to appear together on a single root-to-leaf path.
    /// XGBoost `interaction_constraints`.
    pub interaction_constraints: Vec<Vec<u32>>,
    /// Trees grown per output per round (boosted random forests; in
    /// `1..=65536`). XGBoost `num_parallel_tree`. `gblinear` needs `1`.
    ///
    /// Every tree of a round's forest grows from the same gradients, draws
    /// its own column sample, and has its leaves shrunk by
    /// `eta / num_parallel_tree`. Under `hist` and `exact` each tree also
    /// draws its own row sample; under `approx` the forest shares one, as
    /// XGBoost's approx updater does.
    pub num_parallel_tree: usize,
    /// Row subsampling method. XGBoost `sampling_method`.
    pub sampling_method: SamplingMethod,
    /// LightGBM's class-balanced bagging for binary classification
    /// ([`BalancedBagging`]; `pos_bagging_fraction` /
    /// `neg_bagging_fraction`), `None` (the default) for
    /// off. It replaces `subsample`, which must stay `1` (LightGBM ignores
    /// `bagging_fraction` then), and needs a `binary:*` objective, a tree
    /// booster (a classic `booster = ebm` tree draws from its outer bag),
    /// uniform sampling, and one label column of `0`/`1` labels; Boulevard
    /// inference (`booster = boulevard`, `ebm_boulevard`) refuses it.
    pub balanced_bagging: Option<BalancedBagging>,
    /// LightGBM's query-level bagging for ranking ([`QueryBagging`];
    /// `bagging_by_query`), `None` (the default) for off:
    /// whole query groups are kept or dropped each round. It replaces
    /// `subsample`, which must stay `1`, and needs a `rank:*` objective, a
    /// tree booster (a classic `booster = ebm` tree keeps the rows of its
    /// outer bag in the kept queries), uniform sampling, and query groups on
    /// the training data.
    pub bagging_by_query: Option<QueryBagging>,
    /// Output-to-tree allocation for multi-output models. XGBoost
    /// `multi_strategy`.
    pub multi_strategy: MultiStrategy,
    /// Grow new trees or update existing ones (with the refresh updater's
    /// options). XGBoost `process_type`.
    pub process_type: ProcessType,

    // ---- LightGBM tree options (opt-in) ----
    /// Extremely randomized split search (LightGBM `extra_trees`), `None`
    /// for XGBoost's exhaustive search. Requires the histogram builder
    /// (`hist`/`approx`) and one output per tree; refused with
    /// `grow_policy = symmetric`.
    pub extra_trees: Option<ExtraTrees>,
    /// Path smoothing strength `s >= 0` (LightGBM `path_smooth`, `0` = off).
    /// Each child's output is pulled toward its parent's:
    /// `w = w_raw·(n/s)/(n/s + 1) + w_parent/(n/s + 1)` with `n` the child's
    /// row count, and splits are scored at the smoothed outputs. Requires the
    /// histogram builder (`hist`/`approx`) and one output per tree; refused
    /// with `grow_policy = symmetric`.
    pub path_smooth: f64,
    /// Fit a ridge-regularized linear model in every leaf (LightGBM
    /// `linear_tree`) on the numerical features split on along the leaf's
    /// path; rows with a missing value in any of them predict the constant
    /// leaf value. The first boosting round keeps constant leaves. Requires
    /// the histogram builder (`hist`/`approx`) and one output per tree;
    /// refused with `reg:absoluteerror` and `reg:quantileerror`, whose leaves
    /// are re-estimated after growth. Linear-leaf models use the native
    /// formats only: SHAP, XGBoost export, and the compact format refuse
    /// them.
    pub linear_tree: Option<LinearTree>,
    /// Train on quantized gradients (LightGBM `use_quantized_grad`), `None`
    /// for full precision. Needs `tree_method` `hist`/`approx` (or `auto`)
    /// and a tree booster.
    pub quantized: Option<QuantizedGrad>,

    // ---- Compact training (Trees on a Diet; opt-in) ----
    /// Penalty `ι` subtracted from the loss change of a split on a feature the
    /// ensemble does not use yet (Herrmann et al., *Boosted Trees on a Diet*,
    /// ICLR 2026, eq. 3). Same units as [`gamma`](Self::gamma); `0` (the
    /// default) disables it. Pair with
    /// [`BoostedModel::to_compact_bytes`](crate::model::BoostedModel::to_compact_bytes),
    /// whose dictionaries shrink as features and thresholds are reused. The
    /// paper's `toad_penalty_feature`. Both penalties act in the XGBoost
    /// split searches of every tree method, and are refused with
    /// `extra_trees`, `path_smooth`, `grow_policy = symmetric`, and
    /// `multi_strategy = multi_output_tree`.
    pub toad_penalty_feature: f64,
    /// Penalty `ξ` subtracted from the loss change of a split at a threshold
    /// (or categorical left set) not yet used for its feature anywhere in the
    /// ensemble; a new feature pays both penalties. Same units as
    /// [`gamma`](Self::gamma); `0` (the default) disables it. The paper's
    /// `toad_penalty_threshold`.
    pub toad_penalty_threshold: f64,

    // ---- SGLB and model shrinkage (CatBoost; opt-in) ----
    /// Stochastic Gradient Langevin Boosting (CatBoost `langevin`;
    /// Ustimenko and Prokhorenkova, ICML 2021; see [`Langevin`]), `None`
    /// for off unless [`posterior_sampling`](Self::posterior_sampling) turns
    /// it on. Langevin adds no model shrinkage of its own: set
    /// [`model_shrink`](Self::model_shrink) for it (CatBoost's flat
    /// `langevin=true` defaults to a constant rate `0.001`, which
    /// [`from_xgboost`](Self::from_xgboost) maps to that `model_shrink`).
    ///
    /// Needs `booster = gbtree` with one tree per output and iteration
    /// (`num_parallel_tree = 1`); refused with monotone constraints,
    /// `linear_tree`, `path_smooth`, quantized leaf renewal
    /// ([`QuantizedGrad::renew_leaf`]; all of which the re-estimated leaves
    /// would bypass), gradient-based sampling (whose row probabilities the
    /// noise would distort), and `process_type = update`.
    pub langevin: Option<Langevin>,
    /// Per-iteration model shrinkage (CatBoost `model_shrink_rate` /
    /// `model_shrink_mode`; see [`ModelShrink`]), `None` for none
    /// ([`posterior_sampling`](Self::posterior_sampling) derives its own).
    /// The constant coefficient `1 - rate * eta` must stay positive.
    ///
    /// A shrunk model stores its trees unscaled with the per-iteration
    /// factors and predicts with training's shrink-then-add arithmetic, so
    /// iteration ranges `..k`,
    /// [`slice`](crate::model::BoostedModel::slice)`(..k, 1)`, and early
    /// stopping reproduce the model trained for `k` rounds exactly. Refused
    /// with `dart`, `gblinear`, `process_type = update`, continued training,
    /// and per-row `base_margin`s.
    pub model_shrink: Option<ModelShrink>,
    /// SGLB posterior sampling (CatBoost `posterior_sampling`): Langevin on
    /// with diffusion temperature `N` and constant model shrinkage at rate
    /// `1 / (2N)`, `N` the number of training rows, so the iterates sample
    /// the Bayesian posterior of the ensemble. The basis of
    /// [`predict_virtual_ensembles`](crate::model::BoostedModel::predict_virtual_ensembles)'
    /// knowledge uncertainty. An explicit Langevin temperature or model
    /// shrinkage is refused rather than overridden.
    pub posterior_sampling: bool,
}

impl Default for TrainingParams {
    fn default() -> Self {
        TrainingParams {
            booster: BoosterKind::GbTree,
            nthread: None,
            seed: 0,
            device: Device::Cpu,
            objective: Objective::default(),
            base_score: None,
            eval_metric: Vec::new(),
            eta: 0.3,
            gamma: 0.0,
            max_depth: NonZeroUsize::new(6),
            max_leaves: None,
            min_child_weight: 1.0,
            max_delta_step: MaxDeltaStep::ObjectiveDefault,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            lambda: 1.0,
            alpha: 0.0,
            tree_method: TreeMethod::Auto,
            grow_policy: GrowPolicy::DepthWise,
            max_bin: 256,
            monotone_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            num_parallel_tree: 1,
            sampling_method: SamplingMethod::Uniform,
            bagging_by_query: None,
            balanced_bagging: None,
            multi_strategy: MultiStrategy::OneOutputPerTree,
            process_type: ProcessType::Default,
            extra_trees: None,
            path_smooth: 0.0,
            linear_tree: None,
            quantized: None,
            toad_penalty_feature: 0.0,
            toad_penalty_threshold: 0.0,
            langevin: None,
            model_shrink: None,
            posterior_sampling: false,
        }
    }
}

/// [`ensure`] that `objective` is `reg:squarederror` at `scale_pos_weight =
/// 1`, which `who` (a Boulevard fit, which also refuses sample weights)
/// needs.
fn unweighted_squared_error(objective: &Objective, who: &str) -> Result<()> {
    let got = match objective {
        Objective::SquaredError(r) => format!(
            "`reg:squarederror` at `scale_pos_weight = {}`",
            r.scale_pos_weight()
        ),
        other => format!("`{}`", other.name()),
    };
    ensure(
        "objective",
        objective.is_unweighted_squared_error(),
        format!("{who} supports `reg:squarederror` at `scale_pos_weight = 1` only, got {got}"),
    )
}

impl TrainingParams {
    /// Start a builder for ergonomic, chained configuration.
    ///
    /// To derive a variant of an existing configuration through the same
    /// setters and validation, convert it back into a builder
    /// ([`TrainingParamsBuilder::from`]):
    ///
    /// ```
    /// use hessboost::config::TrainingParamsBuilder;
    /// use hessboost::prelude::*;
    ///
    /// # fn main() -> Result<()> {
    /// let base = TrainingParams::builder().max_depth(3).build()?;
    /// let smoothed = TrainingParamsBuilder::from(base.clone())
    ///     .path_smooth(1.0)
    ///     .build()?;
    /// assert_eq!(smoothed.max_depth, std::num::NonZeroUsize::new(3));
    /// assert_eq!(smoothed.path_smooth, 1.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder() -> TrainingParamsBuilder {
        TrainingParams::default().into()
    }

    /// Validate mutually-consistent ranges. Called automatically before training.
    ///
    /// The checks run in a fixed order (numeric ranges, reuse penalties,
    /// device, objective parameters, booster, tree shape, training modes,
    /// tree options, SGLB and model shrinkage, Boulevard), so a
    /// configuration that breaks several rules always reports the same one.
    pub fn validate(&self) -> Result<()> {
        self.validate_ranges()?;
        self.validate_reuse_penalties()?;
        self.validate_device()?;
        self.validate_objective_params()?;
        ensure(
            "num_parallel_tree",
            (1..=MAX_NUM_PARALLEL_TREE).contains(&self.num_parallel_tree),
            format!(
                "must be in [1, {MAX_NUM_PARALLEL_TREE}], got {}",
                self.num_parallel_tree
            ),
        )?;
        if self.booster == BoosterKind::GbLinear {
            self.validate_gblinear()?;
        }
        self.validate_tree_shape()?;
        self.validate_training_modes()?;
        self.validate_bagging_by_query()?;
        self.validate_balanced_bagging()?;
        self.validate_tree_options()?;
        self.validate_sglb()?;
        self.validate_boulevard()?;
        self.validate_ebm()
    }

    /// Query-level bagging: a ranking objective on a tree booster, with
    /// uniform sampling and no `subsample` it would override.
    fn validate_bagging_by_query(&self) -> Result<()> {
        if self.bagging_by_query.is_none() {
            return Ok(());
        }
        ensure(
            "bagging_by_query",
            self.booster != BoosterKind::GbLinear,
            "query bagging needs a tree booster: `gblinear` samples no rows",
        )?;
        ensure(
            "bagging_by_query",
            self.objective.is_ranking(),
            format!(
                "query bagging needs a `rank:*` objective, not `{}`",
                self.objective.name()
            ),
        )?;
        ensure(
            "bagging_by_query",
            self.balanced_bagging.is_none(),
            "query bagging is not supported together with class-balanced bagging",
        )?;
        ensure(
            "subsample",
            self.subsample == 1.0,
            "query bagging replaces `subsample` with its query fraction; leave it at 1",
        )?;
        ensure(
            "sampling_method",
            self.sampling_method == SamplingMethod::Uniform,
            "query bagging keeps whole queries; `gradient_based` is not supported with it",
        )
    }

    /// Class-balanced bagging: a binary objective on a tree booster, with
    /// uniform sampling and no `subsample` it would override.
    fn validate_balanced_bagging(&self) -> Result<()> {
        if self.balanced_bagging.is_none() {
            return Ok(());
        }
        ensure(
            "pos_bagging_fraction",
            self.booster != BoosterKind::GbLinear,
            "balanced bagging needs a tree booster: `gblinear` samples no rows",
        )?;
        ensure(
            "pos_bagging_fraction",
            self.objective.is_binary_classifier(),
            format!(
                "balanced bagging needs a `binary:*` objective, not `{}`",
                self.objective.name()
            ),
        )?;
        ensure(
            "subsample",
            self.subsample == 1.0,
            "balanced bagging replaces `subsample` (LightGBM ignores \
             `bagging_fraction` then); leave it at 1",
        )?;
        ensure(
            "sampling_method",
            self.sampling_method == SamplingMethod::Uniform,
            "balanced bagging samples uniformly within each class; \
             `gradient_based` is not supported with it",
        )
    }

    /// Whether either reuse penalty (Trees-on-a-Diet) is on.
    fn reuse_penalties_on(&self) -> bool {
        self.toad_penalty_feature > 0.0 || self.toad_penalty_threshold > 0.0
    }

    /// Ranges of the learning rate, regularization, and sampling ratios.
    fn validate_ranges(&self) -> Result<()> {
        positive("eta", self.eta)?;
        narrows("eta", self.eta, true)?;
        non_negative("gamma", self.gamma)?;
        narrows("gamma", self.gamma, false)?;
        non_negative("min_child_weight", self.min_child_weight)?;
        narrows("min_child_weight", self.min_child_weight, false)?;
        if let MaxDeltaStep::Bounded(bound) = self.max_delta_step {
            ensure(
                "max_delta_step",
                bound.is_finite() && bound > 0.0,
                format!("a bound must be > 0 (no bound is `MaxDeltaStep::Unbounded`), got {bound}"),
            )?;
            narrows("max_delta_step", bound, true)?;
        }
        non_negative("lambda", self.lambda)?;
        narrows("lambda", self.lambda, false)?;
        non_negative("alpha", self.alpha)?;
        narrows("alpha", self.alpha, false)?;
        unit("subsample", self.subsample)?;
        // subsample of exactly 0 is meaningless.
        ensure("subsample", self.subsample != 0.0, "must be > 0")?;
        unit("colsample_bytree", self.colsample_bytree)?;
        unit("colsample_bylevel", self.colsample_bylevel)?;
        unit("colsample_bynode", self.colsample_bynode)
    }

    /// Ranges of the reuse penalties and the split searches that apply them.
    fn validate_reuse_penalties(&self) -> Result<()> {
        non_negative("toad_penalty_feature", self.toad_penalty_feature)?;
        non_negative("toad_penalty_threshold", self.toad_penalty_threshold)?;
        ensure(
            "toad_penalty_feature",
            self.booster != BoosterKind::GbLinear
                || (self.toad_penalty_feature == 0.0 && self.toad_penalty_threshold == 0.0),
            "reuse penalties need a tree booster (`gbtree` or `dart`)",
        )?;
        // The penalties act in the XGBoost histogram/exact split searches;
        // the LightGBM split search and symmetric level-wise growth do not
        // apply them, so refuse the combination instead of ignoring it.
        let reuse_on = self.reuse_penalties_on();
        ensure(
            "toad_penalty_feature",
            !(reuse_on
                && (self.extra_trees.is_some()
                    || self.path_smooth > 0.0
                    || self.grow_policy == GrowPolicy::Symmetric)),
            "reuse penalties are not supported with `extra_trees`, `path_smooth`, or \
             `grow_policy=symmetric`",
        )
    }

    /// The GPU backends accelerate the histogram tree method only; the
    /// other tree methods, the quantized path, and `gblinear` have their
    /// own accumulation loops that would silently ignore the device.
    fn validate_device(&self) -> Result<()> {
        let device = self.device;
        match device {
            Device::Cpu => return Ok(()),
            Device::Metal => ensure(
                "device",
                cfg!(all(target_os = "macos", feature = "metal")),
                "`metal` requires building with the `metal` feature on macOS",
            )?,
            Device::Cuda { .. } => ensure(
                "device",
                cfg!(all(target_os = "linux", feature = "cuda")),
                format!("`{device}` requires building with the `cuda` feature on Linux"),
            )?,
        }
        ensure(
            "device",
            !matches!(self.tree_method, TreeMethod::Exact | TreeMethod::Approx),
            format!("`{device}` requires `tree_method = hist` (or `auto`)"),
        )?;
        ensure(
            "device",
            self.quantized.is_none(),
            format!("`{device}` does not support `use_quantized_grad`"),
        )?;
        ensure(
            "device",
            self.booster != BoosterKind::GbLinear,
            format!("`{device}` needs a tree booster (`gbtree` or `dart`)"),
        )?;
        ensure(
            "device",
            !matches!(self.process_type, ProcessType::Update(_)),
            format!("`{device}` does not support `process_type = update` (refresh grows no trees)"),
        )
    }

    /// `base_score` and the objective settings its parameter structs cannot
    /// check alone.
    fn validate_objective_params(&self) -> Result<()> {
        if let Some(base_score) = self.base_score {
            ensure("base_score", base_score.is_finite(), "must be finite")?;
        }
        // A model records a custom loss by its name; a built-in objective's
        // name would reload as that objective, with its transform.
        if let Objective::Custom(loss) = &self.objective {
            ensure(
                "objective",
                !Objective::is_built_in_name(loss.name()),
                format!(
                    "the custom loss is named `{}`, a built-in objective's name, as which a \
                     saved model would reload; rename the loss",
                    loss.name()
                ),
            )?;
        }
        // Only shared (vector-leaf) trees choose their structure from one
        // distribution parameter; other layouts would ignore the direction.
        if let Objective::Dist(dist) = &self.objective {
            ensure(
                "dist_split_direction",
                dist.split_direction().is_none()
                    || self.multi_strategy == MultiStrategy::MultiOutputTree,
                "chooses the structure of shared trees and needs \
                 `multi_strategy=multi_output_tree`",
            )?;
        }
        Ok(())
    }

    /// Histogram bins, tree size bounds, and the symmetric-growth depth.
    fn validate_tree_shape(&self) -> Result<()> {
        ensure(
            "max_bin",
            self.max_bin >= 2,
            format!("must be >= 2, got {}", self.max_bin),
        )?;
        ensure(
            "max_leaves",
            !(self.grow_policy == GrowPolicy::LossGuide
                && self.max_leaves.is_none()
                && self.max_depth.is_none()),
            "lossguide growth needs a bound: set max_leaves or max_depth",
        )?;
        if self.grow_policy == GrowPolicy::Symmetric {
            ensure(
                "grow_policy",
                self.booster != BoosterKind::GbLinear,
                "`symmetric` growth needs a tree booster (`gbtree` or `dart`)",
            )?;
            ensure(
                "max_depth",
                self.max_depth
                    .is_some_and(|depth| depth.get() <= MAX_SYMMETRIC_DEPTH),
                format!(
                    "symmetric growth needs a max_depth in 1..={MAX_SYMMETRIC_DEPTH}, got {}",
                    self.max_depth
                        .map_or_else(|| "no limit".to_owned(), |d| d.to_string())
                ),
            )?;
            ensure(
                "max_leaves",
                self.max_leaves.is_none(),
                "symmetric growth sizes trees by max_depth; leave max_leaves unset",
            )?;
        }
        Ok(())
    }

    /// Compatibility of the vector-leaf and quantized training modes.
    fn validate_training_modes(&self) -> Result<()> {
        if self.multi_strategy == MultiStrategy::MultiOutputTree {
            // The vector-leaf builder has its own (XGBoost) split search:
            // symmetric level-wise growth and the reuse penalties do not
            // reach it.
            ensure(
                "grow_policy",
                self.grow_policy != GrowPolicy::Symmetric,
                "`symmetric` growth is not supported with `multi_strategy=multi_output_tree`",
            )?;
            ensure(
                "toad_penalty_feature",
                !self.reuse_penalties_on(),
                "reuse penalties are not supported with `multi_strategy=multi_output_tree`",
            )?;
        }
        if let Some(quantized) = &self.quantized {
            ensure(
                "use_quantized_grad",
                self.tree_method != TreeMethod::Exact && self.booster != BoosterKind::GbLinear,
                "quantized training needs a tree booster with `tree_method` hist, approx or auto",
            )?;
            ensure(
                "use_quantized_grad",
                self.multi_strategy == MultiStrategy::OneOutputPerTree,
                "quantized training grows one-output trees only",
            )?;
            // Symmetric growth builds its level histograms outside the
            // quantized node path, so the setting would be silently ignored.
            ensure(
                "use_quantized_grad",
                self.grow_policy != GrowPolicy::Symmetric,
                "quantized training is not supported with `grow_policy=symmetric`",
            )?;
            // Path-smoothed leaves keep the outputs their (quantized) splits
            // recorded, so renewed leaf statistics would be discarded.
            ensure(
                "quant_train_renew_leaf",
                !(quantized.renew_leaf() && self.path_smooth > 0.0),
                "leaf renewal is not supported with `path_smooth`",
            )?;
        }
        Ok(())
    }

    /// Refuse the tree-booster settings `gblinear` cannot apply. Coordinate
    /// descent updates every weight from every row each round, grows no
    /// trees, and draws nothing at random, so row and column sampling,
    /// forests, and tree constraints would be silently ignored. Like
    /// XGBoost, which accepts (with an "unused parameter" warning) whatever
    /// tree settings it is given, the tree-shape settings whose defaults are
    /// not neutral (`max_depth`, `min_child_weight`, `max_bin`,
    /// `tree_method`, `grow_policy`, ...) stay accepted: every configuration
    /// carries them. The ones refused here default to "off" and are only
    /// changed to ask for their effect. (The LightGBM and compact-training
    /// tree options are refused by their own checks.)
    fn validate_gblinear(&self) -> Result<()> {
        ensure(
            "num_parallel_tree",
            self.num_parallel_tree == 1,
            "gblinear grows no trees, so it cannot grow forests; must be 1",
        )?;
        ensure(
            "subsample",
            self.subsample == 1.0,
            "gblinear updates from every row and does not subsample; must be 1",
        )?;
        ensure(
            "sampling_method",
            self.sampling_method == SamplingMethod::Uniform,
            "gblinear does not sample rows; `gradient_based` needs a tree booster",
        )?;
        for (name, ratio) in [
            ("colsample_bytree", self.colsample_bytree),
            ("colsample_bylevel", self.colsample_bylevel),
            ("colsample_bynode", self.colsample_bynode),
        ] {
            ensure(
                name,
                ratio == 1.0,
                "gblinear updates every feature and does not sample columns; must be 1",
            )?;
        }
        ensure(
            "monotone_constraints",
            self.monotone_constraints
                .iter()
                .all(|&m| m == Monotone::None),
            "gblinear does not apply monotone constraints",
        )?;
        ensure(
            "interaction_constraints",
            self.interaction_constraints.is_empty(),
            "gblinear does not apply interaction constraints",
        )
    }

    /// Range and compatibility checks of the opt-in LightGBM tree options
    /// ([`extra_trees`](Self::extra_trees), [`path_smooth`](Self::path_smooth),
    /// [`linear_tree`](Self::linear_tree)). They act inside the histogram tree
    /// builder only, so every other booster, builder, or tree layout is
    /// refused instead of silently ignoring them. The split-search options
    /// live in the per-node histogram split search, which symmetric growth
    /// replaces with its level-wise search, so they are refused there too;
    /// linear leaves are fitted after growth and apply to symmetric trees.
    fn validate_tree_options(&self) -> Result<()> {
        non_negative("path_smooth", self.path_smooth)?;
        // The compatibility checks do not depend on the option, so the first
        // enabled one names the error.
        let enabled = [
            ("extra_trees", self.extra_trees.is_some()),
            ("path_smooth", self.path_smooth > 0.0),
            ("linear_tree", self.linear_tree.is_some()),
        ];
        if let Some(&(name, _)) = enabled.iter().find(|&&(_, on)| on) {
            ensure(
                name,
                self.booster != BoosterKind::GbLinear,
                "requires a tree booster (`gbtree` or `dart`)",
            )?;
            ensure(
                name,
                self.tree_method != TreeMethod::Exact,
                "requires the histogram tree builder (`tree_method` `hist`, `approx` or `auto`)",
            )?;
            ensure(
                name,
                self.multi_strategy == MultiStrategy::OneOutputPerTree,
                "is not supported with `multi_strategy=multi_output_tree`",
            )?;
        }
        if let Some(&(name, _)) = enabled[..2].iter().find(|&&(_, on)| on) {
            ensure(
                name,
                self.grow_policy != GrowPolicy::Symmetric,
                "is not supported with `grow_policy=symmetric` (level-wise split search)",
            )?;
        }
        // LightGBM refuses `regression_l1` with linear trees: objectives whose
        // leaves are re-estimated after growth (XGBoost's adaptive leaves)
        // would overwrite the constant that linear leaves fall back to.
        ensure(
            "linear_tree",
            !(self.linear_tree.is_some() && self.objective.has_adaptive_leaves()),
            format!(
                "is not supported with the adaptive-leaf objective `{}`",
                self.objective.name()
            ),
        )
    }

    /// Whether Stochastic Gradient Langevin Boosting is on: set directly or
    /// through [`posterior_sampling`](Self::posterior_sampling).
    pub(crate) fn langevin_on(&self) -> bool {
        self.langevin.is_some() || self.posterior_sampling
    }

    /// The Langevin diffusion temperature in effect for `n_rows` training
    /// rows: the row count under posterior sampling, else the configured
    /// value or CatBoost's `10000`.
    pub(crate) fn effective_diffusion_temperature(&self, n_rows: usize) -> f64 {
        if self.posterior_sampling {
            n_rows as f64
        } else {
            self.langevin
                .and_then(|l| l.diffusion_temperature())
                .unwrap_or(1e4)
        }
    }

    /// The Langevin noise scale `sqrt(2 / (eta * temperature))` (CatBoost's
    /// `CalcLangevinNoiseRate`).
    pub(crate) fn langevin_noise_scale(&self, temperature: f64) -> f64 {
        (2.0 / (self.eta * temperature)).sqrt()
    }

    /// The model shrinkage `(rate, mode)` in effect for `n_rows` training
    /// rows: `1 / (2 n_rows)` constant under posterior sampling, else the
    /// configured shrinkage, if any.
    pub(crate) fn effective_model_shrink(&self, n_rows: usize) -> Option<(f64, ModelShrinkMode)> {
        if self.posterior_sampling {
            return Some((1.0 / (2.0 * n_rows as f64), ModelShrinkMode::Constant));
        }
        self.model_shrink
            .map(|shrink| (shrink.rate(), shrink.mode()))
    }

    /// Whether training shrinks the model every iteration (known without
    /// the data: posterior sampling always shrinks at a positive rate).
    pub(crate) fn model_shrinkage_on(&self) -> bool {
        self.posterior_sampling || self.model_shrink.is_some()
    }

    /// Compatibility of Langevin boosting and model shrinkage (CatBoost's
    /// `TBoostingOptions::Validate` and `TCatBoostOptions::Validate`, plus
    /// what the tree path here supports); the groups validate their own
    /// values.
    fn validate_sglb(&self) -> Result<()> {
        if self.posterior_sampling {
            // CatBoost derives these from the row count and refuses explicit
            // values instead of overriding them.
            ensure(
                "diffusion_temperature",
                self.langevin
                    .is_none_or(|l| l.diffusion_temperature().is_none()),
                "is derived by `posterior_sampling` (the training row count); leave it unset",
            )?;
            ensure(
                "model_shrink_rate",
                self.model_shrink.is_none(),
                "is derived by `posterior_sampling` (constant, 1 / (2 * rows)); leave it unset",
            )?;
        }
        // The noise joins `f32` gradients, so its scale must be a finite,
        // positive `f32`: an underflowing `eta * T` would make every noisy
        // gradient infinite, an overflowing one would switch the noise off
        // (a subnormal scale is tiny but still noise). Under posterior
        // sampling `T` is the row count `n >= 1` and `eta < 2n`
        // (`Sglb::resolve`), so `eta * T` lies in `(2^-150, 2n^2)` and the
        // scale in `(1 / n, 2^76)`: always representable.
        if self.langevin.is_some() && !self.posterior_sampling {
            let temperature = self.effective_diffusion_temperature(0);
            let sigma = self.langevin_noise_scale(temperature);
            ensure(
                "diffusion_temperature",
                (sigma as f32).is_finite() && sigma as f32 > 0.0,
                format!(
                    "gives a Langevin noise scale sqrt(2 / (eta * diffusion_temperature)) \
                     that is not a finite, positive f32: {sigma:e} for eta {:e} and \
                     temperature {temperature:e}",
                    self.eta
                ),
            )?;
        }
        // Posterior sampling's rate depends on the row count; its coefficient
        // is checked with the data (`Sglb::resolve`).
        if let Some(shrink) = self.model_shrink
            && shrink.mode() == ModelShrinkMode::Constant
        {
            ensure(
                "model_shrink_rate",
                shrink.rate() * self.eta < 1.0,
                format!(
                    "the constant shrink coefficient 1 - model_shrink_rate * eta must stay \
                     positive, got rate {} with eta {}",
                    shrink.rate(),
                    self.eta
                ),
            )?;
        }
        let enabled = [
            ("langevin", self.langevin_on()),
            ("model_shrink_rate", self.model_shrinkage_on()),
        ];
        for (name, _) in enabled.iter().filter(|&&(_, on)| on) {
            // DART rescales its trees with the contribution weights that
            // shrinkage stores; gblinear grows no trees; refresh grows
            // nothing new.
            ensure(
                name,
                self.booster == BoosterKind::GbTree,
                "requires `booster = gbtree`",
            )?;
            ensure(
                name,
                self.process_type == ProcessType::Default,
                "is not supported with `process_type = update` (refresh grows no trees)",
            )?;
        }
        if self.langevin_on() {
            // The noise scale assumes one tree carries each output's whole
            // step; the re-estimated leaves would bypass the constraint
            // bounds, the path-smoothed outputs, the leaf linear fits, and
            // quantized training's renewed leaves.
            ensure(
                "langevin",
                self.num_parallel_tree == 1,
                "requires `num_parallel_tree = 1`",
            )?;
            ensure(
                "langevin",
                self.monotone_constraints
                    .iter()
                    .all(|&m| m == Monotone::None),
                "is not supported with monotone constraints",
            )?;
            ensure(
                "langevin",
                self.linear_tree.is_none() && self.path_smooth == 0.0,
                "is not supported with `linear_tree` or `path_smooth`",
            )?;
            ensure(
                "langevin",
                self.quantized.is_none_or(|q| !q.renew_leaf()),
                "is not supported with `quant_train_renew_leaf` (the Langevin leaf \
                 re-estimation would replace the renewed leaves)",
            )?;
            ensure(
                "langevin",
                !(self.sampling_method == SamplingMethod::GradientBased && self.subsample < 1.0),
                "is not supported with `sampling_method = gradient_based` (the noise would \
                 distort its row probabilities)",
            )?;
        }
        Ok(())
    }

    /// Ranges of the Boulevard options, and the settings `booster =
    /// boulevard` refuses. Its inference ([`crate::inference`]) reads every
    /// tree as a linear smoother of the round's residuals (a leaf predicts
    /// `Σ z / (m + lambda)` over its `m` sampled rows), so the options that
    /// make leaf values nonlinear in the labels (L1 leaves, clipped leaves,
    /// monotone clipping, quantized gradients, linear or smoothed leaves),
    /// that reweight rows by their residuals (gradient-based sampling) or
    /// sample them by their labels (class-balanced bagging), or that change
    /// the loss are refused. Structure-only options (depth,
    /// `min_child_weight`, `gamma`, column sampling, `extra_trees`,
    /// interaction constraints, categorical splits) are accepted.
    fn validate_boulevard(&self) -> Result<()> {
        let BoosterKind::Boulevard(boulevard) = self.booster else {
            return Ok(());
        };
        let dropout = boulevard.dropout();
        self.refuse_balanced_bagging()?;
        unweighted_squared_error(&self.objective, "`booster = boulevard`")?;
        if self.num_parallel_tree > 1 {
            ensure(
                "boulevard_dropout",
                dropout == 0.0,
                "BRAT-P (`num_parallel_tree > 1`) leaves one tree per round out instead of \
                 dropping trees at random; must be 0",
            )?;
            ensure(
                "eta",
                self.eta == 1.0,
                format!(
                    "BRAT-P (`num_parallel_tree > 1`) has no learning rate; must be 1, got {}",
                    self.eta
                ),
            )?;
        } else {
            ensure(
                "eta",
                self.eta <= 1.0,
                format!(
                    "Boulevard's learning rate must be in (0, 1], got {}",
                    self.eta
                ),
            )?;
        }
        self.validate_linear_smoother()
    }

    /// Class-balanced bagging under Boulevard inference (of `booster =
    /// boulevard` and of `ebm_boulevard`), checked before the objective:
    /// balanced bagging needs a `binary:*` objective, and the reason it
    /// cannot work is not the loss.
    fn refuse_balanced_bagging(&self) -> Result<()> {
        ensure(
            "pos_bagging_fraction",
            self.balanced_bagging.is_none(),
            "class-balanced bagging keeps a row with a probability set by its label, so a leaf \
             is no longer a linear smoother of the labels; Boulevard needs uniform `subsample`",
        )
    }

    /// The settings Boulevard inference (of `booster = boulevard` and of
    /// `booster = ebm` with `ebm_boulevard`) refuses: every tree must be a
    /// linear smoother of its round's residuals with constant leaves.
    fn validate_linear_smoother(&self) -> Result<()> {
        let nonlinear = "makes leaf values nonlinear in the labels, which Boulevard inference \
                         cannot represent";
        ensure(
            "alpha",
            self.alpha == 0.0,
            format!("L1 regularization {nonlinear}; must be 0"),
        )?;
        ensure(
            "max_delta_step",
            self.effective_max_delta_step() == 0.0,
            format!("clipping leaves {nonlinear}; leave it unbounded"),
        )?;
        ensure(
            "monotone_constraints",
            self.monotone_constraints
                .iter()
                .all(|&m| m == Monotone::None),
            format!("clipping leaves to monotone bounds {nonlinear}"),
        )?;
        ensure(
            "use_quantized_grad",
            self.quantized.is_none(),
            format!("quantized gradients {nonlinear}"),
        )?;
        ensure(
            "linear_tree",
            self.linear_tree.is_none(),
            "linear leaves are not constant smoothers; Boulevard needs constant leaves",
        )?;
        ensure(
            "path_smooth",
            self.path_smooth == 0.0,
            "smoothed leaves mix in their ancestors' rows; must be 0",
        )?;
        ensure(
            "sampling_method",
            self.sampling_method == SamplingMethod::Uniform,
            "gradient-based sampling reweights rows by their residuals; Boulevard needs uniform \
             subsampling",
        )?;
        ensure(
            "process_type",
            self.process_type == ProcessType::Default,
            "`update` refreshes existing trees; Boulevard models are grown in one run",
        )
    }

    /// Ranges of the EBM options, the settings `booster = ebm` refuses
    /// (anything that would let a tree reach features outside its term, or
    /// leaves the shape functions cannot read), and with `ebm_boulevard`
    /// the Boulevard inference refusals.
    fn validate_ebm(&self) -> Result<()> {
        let BoosterKind::Ebm(ebm) = self.booster else {
            return Ok(());
        };
        let term = "`booster = ebm` fixes every tree's features to its term";
        ensure(
            "num_parallel_tree",
            self.num_parallel_tree == 1,
            "`booster = ebm` grows one tree per term at a time; must be 1",
        )?;
        for (name, ratio) in [
            ("colsample_bytree", self.colsample_bytree),
            ("colsample_bylevel", self.colsample_bylevel),
            ("colsample_bynode", self.colsample_bynode),
        ] {
            ensure(name, ratio == 1.0, format!("{term}; must be 1"))?;
        }
        ensure(
            "interaction_constraints",
            self.interaction_constraints.is_empty(),
            format!("{term}; must be empty"),
        )?;
        ensure(
            "linear_tree",
            self.linear_tree.is_none(),
            "EBM shape functions need constant leaves",
        )?;
        ensure(
            "toad_penalty_feature",
            !self.reuse_penalties_on(),
            "reuse penalties are not supported with `booster = ebm`",
        )?;
        ensure(
            "process_type",
            self.process_type == ProcessType::Default,
            "`update` refreshes existing trees; EBM models are grown in one run",
        )?;
        ensure(
            "sampling_method",
            self.sampling_method == SamplingMethod::Uniform,
            "`booster = ebm` samples rows uniformly (`subsample`)",
        )?;
        if !ebm.boulevard() {
            return Ok(());
        }
        self.refuse_balanced_bagging()?;
        unweighted_squared_error(&self.objective, "`ebm_boulevard`")?;
        ensure(
            "eta",
            self.eta <= 1.0,
            format!(
                "Boulevard's learning rate must be in (0, 1], got {}",
                self.eta
            ),
        )?;
        ensure(
            "base_score",
            self.base_score.is_none(),
            "the Boulevard EBM's centered terms leave the label mean as the intercept; leave it \
             unset",
        )?;
        self.validate_linear_smoother()
    }

    /// The `booster = ebm` settings (the defaults for any other booster).
    pub(crate) fn ebm_settings(&self) -> Ebm {
        match self.booster {
            BoosterKind::Ebm(ebm) => ebm,
            _ => Ebm::default(),
        }
    }

    /// The `max_delta_step` in effect (`0` = no bound): the configured
    /// bound, or the objective's default (XGBoost's 0.7 for
    /// `count:poisson`).
    pub(crate) fn effective_max_delta_step(&self) -> f64 {
        self.max_delta_step.resolve(&self.objective)
    }

    /// The loss this configuration trains with on data with `n_targets`
    /// label columns: the custom loss itself, or the built-in objective's
    /// with its parameters, the `max_delta_step` in effect, and (with
    /// `multi_strategy = multi_output_tree`) the shared-tree split of a
    /// `dist:*` objective.
    ///
    /// `reg:squarederror`, `reg:pseudohubererror`, `reg:logistic`,
    /// `binary:logistic`, and `reg:absoluteerror` accept a label matrix and
    /// give one output per label column, as in XGBoost; quantile and
    /// expectile regression give one output per level, `dist:*` one per
    /// distribution parameter, multiclass one per class.
    ///
    /// # Errors
    ///
    /// `n_targets > 1` for an objective that models one target per row
    /// (`invalid parameter "labels"`).
    pub fn loss(&self, n_targets: usize) -> Result<Arc<dyn Loss>> {
        self.objective.build_loss(&LossContext {
            n_targets,
            max_delta_step: self.effective_max_delta_step(),
            shared_tree_seed: (self.multi_strategy == MultiStrategy::MultiOutputTree)
                .then_some(self.seed),
            seed: self.seed,
        })
    }

    /// Refuse every setting of `self` that differs from `allowed` (budget
    /// mode and the refresh updater: `allowed` is the defaults plus what
    /// they read). Every field is compared
    /// ([`TrainingParams::changed_keys`] destructures the whole struct, so a
    /// field added later is covered too). The error names `param` and lists
    /// the XGBoost keys after `reason`: "`reason`; leave `a`, `b` at the
    /// default".
    pub(crate) fn refuse_changes_from(
        &self,
        allowed: &TrainingParams,
        param: &'static str,
        reason: &str,
    ) -> Result<()> {
        let changed: Vec<String> = self
            .changed_keys(allowed)
            .into_iter()
            .map(|key| format!("`{key}`"))
            .collect();
        if changed.is_empty() {
            Ok(())
        } else {
            Err(HessboostError::invalid_param(
                param,
                format!("{reason}; leave {} at the default", changed.join(", ")),
            ))
        }
    }
}

/// Builder for [`TrainingParams`].
///
/// Every setter returns `self` for chaining. Terminal method is
/// [`TrainingParamsBuilder::build`], which validates the configuration and
/// reports a setter's refused value (e.g. `max_depth(0)`) by its key.
#[derive(Debug, Clone)]
pub struct TrainingParamsBuilder {
    params: TrainingParams,
    /// Keys whose setter got a value no field can hold, with the reason
    /// [`build`](Self::build) reports.
    refused: Vec<(&'static str, &'static str)>,
}

impl TrainingParamsBuilder {
    setter!(/// Set the booster kind.
        booster: BoosterKind => params.booster);
    /// Set the number of worker threads. `0` is refused at
    /// [`build`](Self::build); [`global_pool`](Self::global_pool) uses the
    /// global Rayon pool (the default).
    #[must_use]
    pub fn nthread(mut self, threads: usize) -> Self {
        self.params.nthread = self.non_zero(
            "nthread",
            threads,
            "must be >= 1 (`global_pool()` uses the global Rayon pool), got 0",
        );
        self
    }
    /// Train on the global Rayon pool (the default; XGBoost `nthread = 0`).
    #[must_use]
    pub fn global_pool(mut self) -> Self {
        self.forget("nthread");
        self.params.nthread = None;
        self
    }
    setter!(/// Set the RNG seed.
        seed: u64 => params.seed);
    setter!(/// Set the processor training runs on (XGBoost `device`).
        device: Device => params.device);
    setter!(/// Set the learning rate (`eta`).
        eta: f64 => params.eta);
    setter!(/// Set the minimum split loss (`gamma`).
        gamma: f64 => params.gamma);
    /// Set the maximum tree depth. `0` is refused at [`build`](Self::build);
    /// [`unlimited_depth`](Self::unlimited_depth) removes the limit.
    #[must_use]
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.params.max_depth = self.non_zero(
            "max_depth",
            depth,
            "must be >= 1 (`unlimited_depth()` removes the limit), got 0",
        );
        self
    }
    /// Grow trees without a depth limit (XGBoost `max_depth = 0`).
    #[must_use]
    pub fn unlimited_depth(mut self) -> Self {
        self.forget("max_depth");
        self.params.max_depth = None;
        self
    }
    /// Set the maximum number of leaves per `lossguide` (or vector-leaf)
    /// tree ([`TrainingParams::max_leaves`]). `0` is refused at
    /// [`build`](Self::build); [`unlimited_leaves`](Self::unlimited_leaves)
    /// removes the limit (the default).
    #[must_use]
    pub fn max_leaves(mut self, leaves: usize) -> Self {
        self.params.max_leaves = self.non_zero(
            "max_leaves",
            leaves,
            "must be >= 1 (`unlimited_leaves()` removes the limit), got 0",
        );
        self
    }
    /// Grow trees without a leaf limit (the default; XGBoost
    /// `max_leaves = 0`).
    #[must_use]
    pub fn unlimited_leaves(mut self) -> Self {
        self.forget("max_leaves");
        self.params.max_leaves = None;
        self
    }
    setter!(/// Set the minimum child hessian weight.
        min_child_weight: f64 => params.min_child_weight);
    setter!(/// Set the bound on each leaf weight (XGBoost `max_delta_step`).
        max_delta_step: MaxDeltaStep => params.max_delta_step);
    setter!(/// Set the row subsample ratio.
        subsample: f64 => params.subsample);
    setter!(/// Set the per-tree column subsample ratio.
        colsample_bytree: f64 => params.colsample_bytree);
    setter!(/// Set the per-level column subsample ratio.
        colsample_bylevel: f64 => params.colsample_bylevel);
    setter!(/// Set the per-node column subsample ratio.
        colsample_bynode: f64 => params.colsample_bynode);
    setter!(/// Set the L2 regularization (`lambda`).
        lambda: f64 => params.lambda);
    setter!(/// Set the L1 regularization (`alpha`).
        alpha: f64 => params.alpha);
    setter!(/// Set the tree construction method.
        tree_method: TreeMethod => params.tree_method);
    setter!(/// Set the tree growth policy.
        grow_policy: GrowPolicy => params.grow_policy);
    setter!(/// Set the maximum histogram bins per feature.
        max_bin: usize => params.max_bin);
    setter!(/// Set the number of trees grown per output per round (`num_parallel_tree`).
        num_parallel_tree: usize => params.num_parallel_tree);
    setter!(/// Set the row subsampling method (`sampling_method`).
        sampling_method: SamplingMethod => params.sampling_method);
    setter!(/// Set the multi-output tree strategy (`multi_strategy`).
        multi_strategy: MultiStrategy => params.multi_strategy);
    setter!(/// Set whether rounds grow or update trees (`process_type`).
        process_type: ProcessType => params.process_type);
    setter!(/// Set LightGBM's path smoothing strength (`path_smooth`, `0` = off).
        path_smooth: f64 => params.path_smooth);
    setter!(/// Set the new-feature reuse penalty `ι` (`toad_penalty_feature`).
        toad_penalty_feature: f64 => params.toad_penalty_feature);
    setter!(/// Set the new-threshold reuse penalty `ξ` (`toad_penalty_threshold`).
        toad_penalty_threshold: f64 => params.toad_penalty_threshold);

    /// Enable LightGBM's randomized split search (`extra_trees`).
    #[must_use]
    pub fn extra_trees(mut self, extra_trees: ExtraTrees) -> Self {
        self.params.extra_trees = Some(extra_trees);
        self
    }

    /// Enable LightGBM's per-leaf linear models (`linear_tree`).
    #[must_use]
    pub fn linear_tree(mut self, linear_tree: LinearTree) -> Self {
        self.params.linear_tree = Some(linear_tree);
        self
    }

    /// Enable LightGBM's query-level bagging (`bagging_by_query`).
    #[must_use]
    pub fn bagging_by_query(mut self, bagging: QueryBagging) -> Self {
        self.params.bagging_by_query = Some(bagging);
        self
    }
    /// Enable LightGBM's class-balanced bagging (`pos_bagging_fraction`,
    /// `neg_bagging_fraction`).
    #[must_use]
    pub fn balanced_bagging(mut self, bagging: BalancedBagging) -> Self {
        self.params.balanced_bagging = Some(bagging);
        self
    }

    /// Enable quantized-gradient training (LightGBM `use_quantized_grad`).
    #[must_use]
    pub fn quantized(mut self, quantized: QuantizedGrad) -> Self {
        self.params.quantized = Some(quantized);
        self
    }

    /// Enable Stochastic Gradient Langevin Boosting (CatBoost `langevin`).
    #[must_use]
    pub fn langevin(mut self, langevin: Langevin) -> Self {
        self.params.langevin = Some(langevin);
        self
    }

    /// Enable per-iteration model shrinkage (CatBoost `model_shrink_rate`
    /// and `model_shrink_mode`).
    #[must_use]
    pub fn model_shrink(mut self, model_shrink: ModelShrink) -> Self {
        self.params.model_shrink = Some(model_shrink);
        self
    }

    /// Enable SGLB posterior sampling (CatBoost `posterior_sampling`).
    #[must_use]
    pub fn posterior_sampling(mut self, posterior_sampling: bool) -> Self {
        self.params.posterior_sampling = posterior_sampling;
        self
    }

    /// Set the objective (default [`Objective::SquaredError`] at [`RegLoss::default`](crate::objective::RegLoss::default)).
    #[must_use]
    pub fn objective(mut self, objective: Objective) -> Self {
        self.params.objective = objective;
        self
    }

    /// Set the base score / global bias.
    #[must_use]
    pub fn base_score(mut self, v: f64) -> Self {
        self.params.base_score = Some(v);
        self
    }

    /// Add an evaluation metric (evaluated after the ones added before).
    #[must_use]
    pub fn eval_metric(mut self, metric: crate::metric::EvalMetric) -> Self {
        self.params.eval_metric.push(metric);
        self
    }

    setter!(/// Set the per-feature monotone constraints.
        monotone_constraints: Vec<Monotone> => params.monotone_constraints);
    setter!(
        /// Set the allowed feature-interaction groups.
        ///
        /// Each inner vector lists feature indices that are permitted to appear
        /// together on a single root-to-leaf path. An empty list disables the
        /// constraint. Mirrors XGBoost `interaction_constraints`.
        interaction_constraints: Vec<Vec<u32>> => params.interaction_constraints
    );

    /// Drop the refusal recorded for `key`, if any.
    fn forget(&mut self, key: &'static str) {
        self.refused.retain(|&(refused, _)| refused != key);
    }

    /// `value` as a limit, recording `reason` for `key` when it is `0`
    /// (a later setting of the same key replaces the refusal).
    fn non_zero(
        &mut self,
        key: &'static str,
        value: usize,
        reason: &'static str,
    ) -> Option<NonZeroUsize> {
        self.forget(key);
        let limit = NonZeroUsize::new(value);
        if limit.is_none() {
            self.refused.push((key, reason));
        }
        limit
    }

    /// Validate and produce the [`TrainingParams`].
    ///
    /// # Errors
    ///
    /// A value a setter refused (`max_depth(0)`, ...) or one
    /// [`TrainingParams::validate`] refuses, as `invalid parameter` naming
    /// the key.
    pub fn build(self) -> Result<TrainingParams> {
        if let Some(&(key, reason)) = self.refused.first() {
            return Err(HessboostError::invalid_param(key, reason));
        }
        self.params.validate()?;
        Ok(self.params)
    }
}

impl From<TrainingParams> for TrainingParamsBuilder {
    /// A builder starting from `params` (validated again by
    /// [`build`](TrainingParamsBuilder::build)).
    fn from(params: TrainingParams) -> Self {
        TrainingParamsBuilder {
            params,
            refused: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective::RegLoss;

    /// The parameter `builder.build()` rejects, if any.
    fn rejected(builder: TrainingParamsBuilder) -> Option<&'static str> {
        match builder.build() {
            Err(HessboostError::InvalidParameter { name, .. }) => Some(name),
            _ => None,
        }
    }

    #[test]
    fn defaults_match_xgboost() {
        let p = TrainingParams::default();
        assert_eq!(p.eta, 0.3);
        assert_eq!(p.max_depth, NonZeroUsize::new(6));
        assert_eq!(p.max_leaves, None);
        assert_eq!(p.nthread, None);
        assert_eq!(p.max_delta_step, MaxDeltaStep::ObjectiveDefault);
        assert_eq!(p.min_child_weight, 1.0);
        assert_eq!(p.lambda, 1.0);
        assert_eq!(p.alpha, 0.0);
        assert_eq!(p.max_bin, 256);
        assert_eq!(p.booster, BoosterKind::GbTree);
        assert_eq!(p.grow_policy, GrowPolicy::DepthWise);
        assert!(p.base_score.is_none());
        assert_eq!(p.objective, Objective::SquaredError(RegLoss::default()));
        p.validate().unwrap();
    }

    #[test]
    fn builder_chains_and_validates() {
        let p = TrainingParams::builder()
            .objective(Objective::BinaryLogistic(
                crate::objective::RegLoss::default(),
            ))
            .eta(0.1)
            .max_depth(4)
            .subsample(0.8)
            .lambda(2.0)
            .build()
            .unwrap();
        assert_eq!(p.objective.name(), "binary:logistic");
        assert_eq!(p.eta, 0.1);
        assert_eq!(p.max_depth, NonZeroUsize::new(4));
        assert_eq!(p.subsample, 0.8);
    }

    #[test]
    fn rejects_bad_params() {
        let b = TrainingParams::builder;
        for (name, builder) in [
            ("eta", b().eta(0.0)),
            ("subsample", b().subsample(1.5)),
            ("lambda", b().lambda(-1.0)),
            ("max_bin", b().max_bin(1)),
            (
                "max_delta_step",
                b().max_delta_step(MaxDeltaStep::Bounded(-1.0)),
            ),
            // No bound is `Unbounded`, not a zero bound.
            (
                "max_delta_step",
                b().max_delta_step(MaxDeltaStep::Bounded(0.0)),
            ),
            (
                "max_delta_step",
                b().max_delta_step(MaxDeltaStep::Bounded(f64::NAN)),
            ),
            ("num_parallel_tree", b().num_parallel_tree(0)),
            // Would overflow the iteration's allocations.
            ("num_parallel_tree", b().num_parallel_tree(1 << 63)),
            (
                "num_parallel_tree",
                b().num_parallel_tree(MAX_NUM_PARALLEL_TREE + 1),
            ),
            // Finite in f64, but infinite or zero in the f32 the split
            // search and objectives use.
            ("eta", b().eta(1e39)),
            ("eta", b().eta(1e-50)),
            ("gamma", b().gamma(1e39)),
            ("min_child_weight", b().min_child_weight(1e39)),
            ("lambda", b().lambda(1e39)),
            ("alpha", b().alpha(1e39)),
            (
                "max_delta_step",
                b().max_delta_step(MaxDeltaStep::Bounded(1e39)),
            ),
            (
                "max_delta_step",
                b().max_delta_step(MaxDeltaStep::Bounded(1e-50)),
            ),
            // Lossguide growth needs a leaf or depth bound.
            (
                "max_leaves",
                b().grow_policy(GrowPolicy::LossGuide).unlimited_depth(),
            ),
        ] {
            assert_eq!(rejected(builder), Some(name));
        }
        assert!(
            b().grow_policy(GrowPolicy::LossGuide)
                .max_leaves(31)
                .build()
                .is_ok()
        );
    }

    /// A zero limit is refused by `build` under its key, the last setting
    /// of a key wins, and the `unlimited_*` / `global_pool` setters are the
    /// way to lift a limit.
    #[test]
    fn zero_limits_are_refused_until_replaced() {
        let b = TrainingParams::builder;
        for (name, builder) in [
            ("max_depth", b().max_depth(0)),
            ("max_leaves", b().max_leaves(0)),
            ("nthread", b().nthread(0)),
            // A refusal is reported even if validation would fail too.
            ("max_depth", b().max_depth(0).eta(0.0)),
        ] {
            assert_eq!(rejected(builder), Some(name));
        }
        let fixed = b().max_depth(0).max_depth(3).nthread(0).global_pool();
        let p = fixed.build().unwrap();
        assert_eq!((p.max_depth, p.nthread), (NonZeroUsize::new(3), None));
        let lifted = b()
            .max_depth(0)
            .unlimited_depth()
            .max_leaves(0)
            .max_leaves(8);
        let p = lifted.build().unwrap();
        assert_eq!((p.max_depth, p.max_leaves), (None, NonZeroUsize::new(8)));
        // A later zero replaces a valid setting.
        assert_eq!(rejected(b().max_depth(4).max_depth(0)), Some("max_depth"));
    }

    /// XGBoost injects `max_delta_step = 0.7` for `count:poisson` only when
    /// the user did not set it; an explicit `0` (`Unbounded`) disables the
    /// constraint, and a bound replaces the default.
    #[test]
    fn poisson_delta_step_default_respects_explicit_zero() {
        let poisson = |step| {
            TrainingParams::builder()
                .objective(Objective::Poisson)
                .max_delta_step(step)
                .build()
                .unwrap()
                .effective_max_delta_step()
        };
        assert_eq!(poisson(MaxDeltaStep::ObjectiveDefault), 0.7);
        assert_eq!(poisson(MaxDeltaStep::Unbounded), 0.0);
        assert_eq!(poisson(MaxDeltaStep::Bounded(0.3)), 0.3);
        assert_eq!(TrainingParams::default().effective_max_delta_step(), 0.0);
    }

    /// A `dist:*` split direction chooses the structure of shared trees
    /// only; other tree layouts would ignore it.
    #[test]
    fn dist_split_direction_needs_shared_trees() {
        use crate::objective::distributional::{DistFamily, DistSplitDirection, Distributional};
        let cyclic = Objective::Dist(
            Distributional::new(DistFamily::Normal)
                .with_split_direction(DistSplitDirection::Cyclic),
        );
        let b = || TrainingParams::builder().objective(cyclic.clone());
        assert_eq!(rejected(b()), Some("dist_split_direction"));
        assert!(
            b().multi_strategy(MultiStrategy::MultiOutputTree)
                .build()
                .is_ok()
        );
    }

    /// Adaptive-leaf objectives re-estimate their leaves after growth, which
    /// would overwrite the constants linear leaves fall back to.
    #[test]
    fn linear_leaves_refuse_adaptive_leaf_objectives() {
        let linear = || TrainingParams::builder().linear_tree(LinearTree::default());
        assert_eq!(
            rejected(linear().objective(Objective::AbsoluteError)),
            Some("linear_tree")
        );
        assert!(
            linear()
                .objective(Objective::Gamma(RegLoss::default()))
                .build()
                .is_ok()
        );
    }

    /// Reuse penalties apply in the XGBoost split searches only; the LightGBM
    /// split search and symmetric growth would silently ignore them.
    #[test]
    fn reuse_penalties_refuse_searches_that_ignore_them() {
        let toad = || TrainingParams::builder().toad_penalty_feature(1.0);
        for params in [
            toad().extra_trees(ExtraTrees::default()),
            toad().path_smooth(1.0),
            toad().grow_policy(GrowPolicy::Symmetric).max_depth(3),
        ] {
            assert_eq!(rejected(params), Some("toad_penalty_feature"));
        }
        assert!(toad().linear_tree(LinearTree::default()).build().is_ok());
    }

    /// Symmetric growth builds histograms outside the quantized path, and
    /// path-smoothed leaves would discard renewed leaf statistics.
    #[test]
    fn quantized_training_refuses_options_it_would_ignore() {
        let q = || {
            TrainingParams::builder()
                .quantized(QuantizedGrad::default())
                .max_depth(3)
        };
        let renewed = QuantizedGrad::builder().renew_leaf(true).build().unwrap();
        assert_eq!(
            rejected(q().grow_policy(GrowPolicy::Symmetric)),
            Some("use_quantized_grad")
        );
        assert!(q().grow_policy(GrowPolicy::LossGuide).build().is_ok());
        // Leaf renewal would be discarded: path-smoothed leaves keep the
        // outputs their quantized splits recorded.
        assert_eq!(
            rejected(q().quantized(renewed).path_smooth(1.0)),
            Some("quant_train_renew_leaf")
        );
        assert!(q().quantized(renewed).build().is_ok());
        assert!(q().path_smooth(1.0).build().is_ok());
    }

    /// The linear booster never grows trees, so symmetric growth would be
    /// silently discarded.
    #[test]
    fn symmetric_growth_refuses_the_linear_booster() {
        let sym = || {
            TrainingParams::builder()
                .grow_policy(GrowPolicy::Symmetric)
                .max_depth(3)
        };
        assert_eq!(
            rejected(sym().booster(BoosterKind::GbLinear)),
            Some("grow_policy")
        );
        assert!(
            sym()
                .booster(BoosterKind::Dart(Dart::default()))
                .build()
                .is_ok()
        );
    }

    /// Coordinate descent reads every row and feature and grows no trees:
    /// sampling, forests, and tree constraints would be silently ignored,
    /// while the tree-shape settings every configuration carries pass.
    #[test]
    fn gblinear_refuses_tree_sampling_forests_and_constraints() {
        let linear = || TrainingParams::builder().booster(BoosterKind::GbLinear);
        for (name, builder) in [
            ("num_parallel_tree", linear().num_parallel_tree(2)),
            ("subsample", linear().subsample(0.5)),
            (
                "sampling_method",
                linear().sampling_method(SamplingMethod::GradientBased),
            ),
            ("colsample_bytree", linear().colsample_bytree(0.5)),
            ("colsample_bylevel", linear().colsample_bylevel(0.5)),
            ("colsample_bynode", linear().colsample_bynode(0.5)),
            (
                "monotone_constraints",
                linear().monotone_constraints(vec![Monotone::None, Monotone::Increasing]),
            ),
            (
                "interaction_constraints",
                linear().interaction_constraints(vec![vec![0, 1]]),
            ),
        ] {
            assert_eq!(rejected(builder), Some(name));
        }
        linear()
            .max_depth(4)
            .min_child_weight(3.0)
            .max_bin(64)
            .tree_method(TreeMethod::Hist)
            .grow_policy(GrowPolicy::LossGuide)
            .monotone_constraints(vec![Monotone::None])
            .build()
            .unwrap();
        for booster in [BoosterKind::GbTree, BoosterKind::Dart(Dart::default())] {
            TrainingParams::builder()
                .booster(booster)
                .num_parallel_tree(2)
                .subsample(0.5)
                .colsample_bynode(0.5)
                .build()
                .unwrap();
        }
    }
}
