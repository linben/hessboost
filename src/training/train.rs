//! The boosting loop: validation, the booster dispatch, and the tree rounds.

use super::api::{TrainResult, Trainer};
use super::eval::{EvalPlan, EvalSet, RoundReporter};
use super::margins::MarginCaches;
use super::prepare::{Prepared, TrainContext, prepare_builder};
use super::round::{RoundPlan, RoundState, grow_round, refresh_round};
use super::row_sampling::RowMeta;
use super::validate::{TrainRequest, validate_request};
use crate::config::{BoosterKind, ProcessType, TrainingParams};
use crate::data::{DMatrix, MetaInfo};
use crate::error::{HessboostError, Result};
use crate::metric::Metric;
use crate::model::{BoostedModel, ModelSpec, Shrinkage};
use crate::objective::{GradPair, Loss};
use crate::training::continuation::{require_model_for_update, resume_model};
use crate::training::multi_output;
use crate::training::sglb::{Sglb, Shrink};
use crate::tree::builder::all_rows;
use crate::tree::reuse::ReuseSet;
use std::num::NonZeroUsize;

/// Run `train` on a dedicated pool of `params.nthread` threads, or on the
/// global rayon pool when `nthread` is unset.
pub(super) fn with_thread_pool<T: Send>(
    params: &TrainingParams,
    train: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    let Some(threads) = params.nthread else {
        return train();
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.get())
        .build()
        .map_err(|error| HessboostError::invalid_param("nthread", error.to_string()))?;
    pool.install(train)
}

/// The core boosting loop, generic over single- and multi-output objectives.
///
/// Margins and gradients are laid out `[instance][output]`. Each round computes
/// all gradients, then grows `num_parallel_tree` trees per output from that
/// output's gradient slice. This is the multi-output generalization of
/// gradient boosting used by multiclass. `init_model` continues training
/// from an existing model ([`Trainer::init_model`]). `objective` is the
/// trainer's own or the one `params` name.
pub(super) fn train_impl(trainer: Trainer<'_>, objective: &dyn Loss) -> Result<TrainResult> {
    let Trainer {
        params,
        dtrain,
        num_boost_round,
        evals,
        early_stopping_rounds,
        metric: metric_override,
        init_model,
        on_round,
    } = trainer;
    let request = TrainRequest {
        params,
        dtrain,
        evals: &evals,
        early_stopping_rounds,
    };
    validate_request(&request, objective)?;
    let info = dtrain.info();
    let n_out = objective.n_outputs();
    let sglb = Sglb::resolve(params, dtrain.n_rows())?;
    let intercepts = || initial_intercepts(params, objective, &info, n_out);
    let model = if let Some(init) = init_model {
        resume_model(init, params, objective, dtrain, num_boost_round, intercepts)?
    } else {
        require_model_for_update(params)?;
        new_model(params, objective, dtrain, intercepts()?)
    };
    let report = RoundReporter::new(on_round);

    if params.booster == BoosterKind::GbLinear {
        return train_linear(&request, num_boost_round, objective, model, report);
    }
    let run = Run {
        ctx: TrainContext {
            params,
            dtrain,
            info: &info,
            objective,
            langevin: sglb.langevin.as_ref(),
            rows: RowMeta::of(dtrain, params),
        },
        evals: &evals,
        early_stopping_rounds,
        metric_override,
        num_boost_round,
        shrink: sglb.shrink.as_ref(),
    };
    match params.booster {
        BoosterKind::Boulevard(_) => train_boulevard(run, model, report),
        BoosterKind::Ebm(_) => train_ebm(run, model, report),
        _ => train_trees(run, model, report),
    }
}

/// A validated tree-booster run past the `gblinear` dispatch: what every
/// round reads, the eval sets, and the settings the loops consume once.
pub(super) struct Run<'a> {
    ctx: TrainContext<'a>,
    evals: &'a [EvalSet<'a>],
    early_stopping_rounds: Option<NonZeroUsize>,
    metric_override: Option<Box<dyn Metric>>,
    num_boost_round: usize,
    /// The per-iteration model shrinkage (SGLB), when configured.
    shrink: Option<&'a Shrink>,
}

impl<'a> Run<'a> {
    /// The metrics every eval set reports ([`EvalPlan::new`], which refuses
    /// the ones that cannot read the eval sets).
    fn eval_plan(&mut self) -> Result<EvalPlan<'a>> {
        let TrainContext {
            params,
            dtrain,
            objective,
            ..
        } = self.ctx;
        EvalPlan::new(
            params,
            objective,
            self.metric_override.take(),
            self.evals,
            dtrain.n_targets(),
        )
    }

    /// The builder state for `tree_method` ([`prepare_builder`]).
    fn prepare(&self) -> Result<Prepared> {
        let TrainContext {
            params,
            dtrain,
            objective,
            ..
        } = self.ctx;
        prepare_builder(params, dtrain, objective.const_hess())
    }
}

/// `booster = boulevard`: `validate` refuses `process_type = update` and
/// continued training for it, and `validate_request` early stopping, so
/// every round grows trees and the history is the whole run's.
fn train_boulevard<'a>(
    mut run: Run<'a>,
    mut model: BoostedModel,
    mut report: RoundReporter<'a>,
) -> Result<TrainResult> {
    let prepared = run.prepare()?;
    let TrainContext { params, dtrain, .. } = run.ctx;
    let mut reuse = ReuseSet::from_params(params, dtrain.n_cols(), model.trees());
    let mut margins = MarginCaches::new(&model, dtrain, run.evals);
    report.watch(run.eval_plan()?, run.early_stopping_rounds, 0);
    let boost = super::boulevard::BoostState {
        model: &mut model,
        margins: &mut margins,
        reuse: &mut reuse,
    };
    super::boulevard::boost(
        &run.ctx,
        &prepared,
        boost,
        run.num_boost_round,
        |round, margins| report.finish_round(round, Some(margins)),
    )?;
    Ok(report.into_result(model))
}

/// `booster = ebm`: `validate` refuses `process_type = update` and
/// continued training for it, and `validate_request` eval sets and early
/// stopping.
fn train_ebm<'a>(
    mut run: Run<'a>,
    mut model: BoostedModel,
    mut report: RoundReporter<'a>,
) -> Result<TrainResult> {
    let prepared = run.prepare()?;
    let eval_plan = run.eval_plan()?;
    super::ebm::boost(
        &run.ctx,
        &prepared,
        &mut model,
        run.num_boost_round,
        // The early-stopping metric: `Trainer::custom_metric`'s, else the
        // last configured one.
        eval_plan.metrics.last().map(AsRef::as_ref),
        &mut |iteration| report.finish_round(iteration, None),
    )?;
    Ok(report.into_result(model))
}

/// gbtree, DART, and forests (and `process_type=update`'s refresh): grow or
/// refresh one iteration per round, then report it.
fn train_trees<'a>(
    mut run: Run<'a>,
    mut model: BoostedModel,
    mut report: RoundReporter<'a>,
) -> Result<TrainResult> {
    let TrainContext {
        params,
        dtrain,
        objective,
        ..
    } = run.ctx;
    let n = dtrain.n_rows();
    let n_out = objective.n_outputs();
    // `process_type=update` refreshes the model's own trees (re-appended one
    // iteration per round) instead of growing new ones, so it needs no
    // builder state.
    let mut plan = match params.process_type {
        ProcessType::Update(refresh) => RoundPlan::Refresh(model.take_trees(), refresh),
        _ => RoundPlan::Grow(run.prepare()?),
    };
    // Continued training numbers its rounds after the model's iterations, so
    // the per-round RNG streams continue where the earlier run stopped.
    let start_iteration = model.num_boost_rounds();
    // Opt-in reuse penalties: the features and thresholds the ensemble already
    // uses, extended by every tree the loop grows. `None` on the default path.
    let reuse = ReuseSet::from_params(params, dtrain.n_cols(), model.trees());
    let margins = MarginCaches::new(&model, dtrain, run.evals);
    report.watch(run.eval_plan()?, run.early_stopping_rounds, start_iteration);
    let mut state = RoundState {
        model,
        margins,
        gpair: vec![GradPair::default(); n * n_out],
        // Per-output gradient scratch; single-output objectives read `gpair`
        // itself ([`gather_output`]).
        gpair_k: if n_out > 1 {
            vec![GradPair::default(); n]
        } else {
            Vec::new()
        },
        reuse,
        noisy_gpair: Vec::new(),
        all_rows: all_rows(n),
        device_margins: false,
    };
    if start_iteration > 0
        && let RoundPlan::Grow(prepared) = &plan
    {
        prepared.resume_approx_cache(
            &run.ctx,
            &state.model.margin_from_trees(dtrain, 0..0),
            &mut state.gpair,
            &mut state.gpair_k,
            n_out,
        )?;
    }
    // `multi_strategy = multi_output_tree` grows vector-leaf trees when there
    // is more than one output (a single output keeps scalar trees, as
    // XGBoost's `LeafLength` does).
    let vector_leaf = multi_output::vector_leaf(params, n_out);
    // Model shrinkage: the intercepts before any shrinkage and every
    // iteration's coefficient (continued training is refused with it, so
    // iterations count from 0).
    let unshrunk_base = state.model.base_scores().to_vec();
    let mut shrink_factors = Vec::new();

    for round in 0..run.num_boost_round {
        let iteration = start_iteration + round;
        if let Some(shrink) = run.shrink {
            let factor = shrink.factor(iteration);
            if factor != 1.0 {
                state.margins.scale(factor);
            }
            shrink_factors.push(factor);
        }
        match &mut plan {
            RoundPlan::Grow(Prepared::Hist { index: ghist, .. }) if vector_leaf => {
                multi_output::boost_round(
                    &multi_output::VectorRound {
                        run: run.ctx,
                        ghist,
                        all_rows: &state.all_rows,
                    },
                    &mut state.model,
                    iteration,
                    &mut state.margins,
                    &mut state.gpair,
                    &mut state.noisy_gpair,
                )?;
            }
            RoundPlan::Refresh(queue, refresh) => {
                refresh_round(&run.ctx, queue, *refresh, iteration, &mut state)?;
            }
            RoundPlan::Grow(prepared) => grow_round(&run.ctx, prepared, iteration, &mut state)?,
        }
        if report
            .finish_round(iteration, Some(&state.margins))
            .is_break()
        {
            break;
        }
    }

    let mut model = state.model;
    if run.shrink.is_some() {
        model.set_shrinkage(Shrinkage::new(shrink_factors, unshrunk_base));
    }
    Ok(report.into_result(model))
}

/// The linear (`gblinear`) booster: fit `model`'s coordinate-descent linear
/// model instead of growing trees, continuing from its weights and margins,
/// reporting each round to `report`. Eval sets and early stopping are
/// refused (the history stays empty).
fn train_linear(
    request: &TrainRequest,
    num_boost_round: usize,
    objective: &dyn Loss,
    mut model: BoostedModel,
    mut report: RoundReporter,
) -> Result<TrainResult> {
    let &TrainRequest {
        params,
        dtrain,
        evals,
        early_stopping_rounds,
    } = request;
    if !evals.is_empty() || early_stopping_rounds.is_some() {
        return Err(HessboostError::invalid_param(
            "booster",
            "gblinear does not yet support evaluation sets or early stopping",
        ));
    }
    let linear = crate::training::gblinear::train_gblinear(
        params,
        dtrain,
        num_boost_round,
        model.margin_from_trees(dtrain, 0..0),
        objective,
        model.linear(),
        &mut |iteration| report.finish_round(iteration, None),
    )?;
    model.set_linear(linear);
    Ok(report.into_result(model))
}

/// An empty model that `loss` (built from `params.objective`) trains on
/// `dtrain`, starting from the margin-space intercepts `base_margins`. The
/// model records `params`' objective (a custom loss by its name), the
/// `max_delta_step` in effect, and the loss's output count, which decides
/// the tree layout (a custom loss's outputs included).
pub(super) fn new_model(
    params: &TrainingParams,
    loss: &dyn crate::objective::Loss,
    dtrain: &DMatrix,
    base_margins: Vec<f32>,
) -> BoostedModel {
    let mut model = BoostedModel::new(
        base_margins,
        ModelSpec {
            objective: crate::model::ModelObjective::trained_with(&params.objective),
            max_delta_step: params.effective_max_delta_step(),
            num_class: params.objective.num_class().unwrap_or(0),
            n_outputs: loss.n_outputs(),
            n_targets: dtrain.n_targets(),
            n_features: dtrain.n_cols(),
        },
    );
    model.set_num_parallel_tree(params.num_parallel_tree);
    model
}

/// Per-output intercepts in margin space. A user-supplied `base_score` is
/// given in prediction space, checked against `loss`'s output domain
/// ([`Loss::validate_base_score`](crate::objective::Loss::validate_base_score)),
/// and broadcast to every output through its link (XGBoost `ProbToMargin`;
/// for multiclass this is a uniform nonzero margin, as in XGBoost);
/// otherwise the loss estimates them from the labels (XGBoost
/// `InitEstimation`). Both go through the loss being trained, never the
/// configured objective's name.
pub(super) fn initial_intercepts(
    params: &TrainingParams,
    loss: &dyn crate::objective::Loss,
    info: &MetaInfo,
    n_out: usize,
) -> Result<Vec<f32>> {
    let base_margins = match params.base_score {
        Some(bs) => {
            loss.validate_base_score(bs)?;
            let mut scores = vec![bs as f32; n_out];
            loss.probs_to_margins(&mut scores);
            scores
        }
        None => loss.base_margins_info(info),
    };
    if base_margins.len() != n_out {
        return Err(HessboostError::dimension_mismatch(
            "objective base_margins length",
            n_out,
            base_margins.len(),
        ));
    }
    if base_margins.iter().any(|m| !m.is_finite()) {
        return Err(HessboostError::invalid_data(
            "labels",
            format!("estimated intercept is not finite ({base_margins:?}); check the labels"),
        ));
    }
    Ok(base_margins)
}

#[cfg(test)]
mod tests;
