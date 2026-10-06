//! One boosting round: gradients, row subsets, and the trees of an iteration
//! (or `process_type=update`'s refresh of them).

use super::dart::{dart_new_tree_weight, finish_dart, round_gradients, round_rng, round_salt};
use super::margins::{MarginCaches, TreeOutput};
use super::prepare::{Prepared, TrainContext, TreeSample, approx_index};
use super::row_sampling::{gradient_sampling, iteration_row_subsets, make_column_sampler};
use crate::config::{BoosterKind, Device, Refresh, SamplingMethod, TrainingParams};
use crate::data::ghist::GHistIndex;
use crate::error::Result;
use crate::model::BoostedModel;
use crate::objective::{GradPair, MIN_HESS, Objective};
use crate::rng::Rng;
use crate::training::multi_output;
use crate::training::refresh::refresh_tree;
use crate::training::sampling::{GradientSample, gradient_based_sample};
use crate::training::sglb::LeafRenewal;
use crate::tree::RegTree;
use crate::tree::builder::{HistTreeBuilder, LeafRows};
use crate::tree::hist::{DeviceLoss, Segment};
use crate::tree::reuse::ReuseSet;
use crate::tree::sampler::ColumnSampler;
use rayon::prelude::*;
use std::sync::OnceLock;

/// What the tree-growing and refresh rounds update: the ensemble, its
/// margin caches, the gradient buffers, and the reuse dictionary.
pub(super) struct RoundState<'a> {
    pub(super) model: BoostedModel,
    pub(super) margins: MarginCaches<'a>,
    /// Every output's gradients, `[row][n_out]`.
    pub(super) gpair: Vec<GradPair>,
    /// One output's gradients gathered from `gpair` (empty for
    /// single-output objectives, which read `gpair` directly).
    pub(super) gpair_k: Vec<GradPair>,
    pub(super) reuse: Option<ReuseSet>,
    /// SGLB's noisy structure gradients, `[row][n_out]` (empty otherwise).
    pub(super) noisy_gpair: Vec<GradPair>,
    /// Every training row, ascending: the rows of an unsampled round.
    pub(super) all_rows: Vec<u32>,
    /// Whether a GPU holds the current training margins (device-resident
    /// rounds), so `margins.train` is stale until read back.
    pub(super) device_margins: bool,
}

/// A device-resident round ([`device_round`]) applies: the GPU computes
/// the gradients from its own margins, grows the tree with the rows on the
/// device, and adds the leaves to its margins, so nothing per row crosses
/// the bus. Only for configurations whose every step the device reproduces
/// bit for bit: `reg:squarederror` or a logistic objective (when the host
/// computes its gradients with a vector kernel) on one label column, one
/// tree per iteration of a plain `gbtree`, every row in every tree, and
/// none of the options that read the host's gradients or rows (SGLB,
/// linear leaves, reuse penalties, model shrinkage).
fn device_round_applies(run: &TrainContext, state: &RoundState) -> Option<DeviceLoss> {
    let TrainContext {
        params,
        dtrain,
        objective,
        langevin,
        ..
    } = *run;
    let plain = objective.n_outputs() == 1
        && dtrain.n_targets() == 1
        && params.num_parallel_tree == 1
        && params.booster == BoosterKind::GbTree
        && params.sampling_method == SamplingMethod::Uniform
        && params.subsample >= 1.0
        && params.balanced_bagging.is_none()
        && params.bagging_by_query.is_none()
        && params.linear_tree.is_none()
        && params.model_shrink.is_none()
        && langevin.is_none()
        && state.reuse.is_none();
    if !plain {
        return None;
    }
    // The scale `Objective::build_loss` gives the loss.
    match &params.objective {
        Objective::SquaredError(loss) => Some(DeviceLoss::SquaredError {
            scale_pos_weight: loss.scale_pos_weight() as f32,
        }),
        Objective::RegLogistic(loss)
        | Objective::BinaryLogistic(loss)
        | Objective::BinaryLogitRaw(loss) => Some(DeviceLoss::Logistic {
            scale_pos_weight: loss.scale_pos_weight() as f32,
            min_hess: MIN_HESS,
            split: crate::simd::logistic_vector_split(dtrain.n_rows())?,
        }),
        _ => None,
    }
}

/// Grow iteration `iteration` as a device-resident round when it applies
/// and the device succeeds; `None` leaves the round to the host path (which
/// first brings the host margins up to date).
fn device_round(
    run: &TrainContext,
    prepared: &Prepared,
    iteration: usize,
    state: &mut RoundState,
) -> Option<()> {
    let loss = device_round_applies(run, state)?;
    let TrainContext {
        params,
        dtrain,
        info,
        ..
    } = *run;
    let (index, backend) = prepared.device_backend()?;
    let engine = backend.row_engine()?;
    if !state.device_margins {
        engine.load_margins(&state.margins.train)?;
        state.device_margins = true;
    }
    // The host's gradients (`gradient_info_at` of the loss), on the device;
    // non-finite ones, and rows only the host reproduces, grow on the host.
    if !engine.gradients(loss, info.label_values(), info.weights)? {
        return None;
    }
    // The host round's RNG draws: no dropout, no row sample, then the
    // tree's column sampler (and no quantization seed).
    let mut rng = round_rng(params, iteration, round_salt(params));
    let mut sampler = make_column_sampler(dtrain, params, &mut rng);
    let builder = HistTreeBuilder::new(params).with_backend(backend);
    let (mut tree, leaves) = builder.build_device_staged(index, &state.all_rows, &mut sampler)?;
    tree.scale_leaves(tree_eta(params));
    let values: Vec<(Segment, f32)> = leaves
        .iter()
        .map(|&(node, seg)| (seg, tree.node(node).leaf_value))
        .collect();
    engine.add_leaf_values(&values)?;
    state
        .margins
        .add_tree_to_evals(&tree, TreeOutput::Scalar(0));
    state
        .model
        .push_tree_weighted(tree, dart_new_tree_weight(None, params));
    Some(())
}

/// Bring `state.margins.train` up to date when the device holds the
/// current margins: read them back, or (after a device failure) recompute
/// them from the model, which reproduces the incremental sums.
fn sync_host_margins(prepared: &Prepared, state: &mut RoundState) {
    if !state.device_margins {
        return;
    }
    state.device_margins = false;
    let read = prepared
        .device_backend()
        .and_then(|(_, backend)| backend.row_engine())
        .and_then(|engine| engine.read_margins(&mut state.margins.train));
    if read.is_none() {
        state.margins.recompute_train(&state.model);
    }
}

/// `process_type=update`: refresh iteration `iteration`'s trees of `queue`
/// output by output with `refresh`'s options, from the gradients of the
/// already refreshed ones, and re-append them.
pub(super) fn refresh_round(
    run: &TrainContext,
    queue: &mut [RegTree],
    refresh: Refresh,
    iteration: usize,
    state: &mut RoundState,
) -> Result<()> {
    let TrainContext {
        params,
        dtrain,
        info,
        objective,
        ..
    } = *run;
    let n_out = objective.n_outputs();
    let parallel = params.num_parallel_tree;
    // Gradients from the already refreshed iterations; iteration `i`'s trees
    // are then refreshed in place, output by output.
    objective.gradient_info_at(&state.margins.train, info, &mut state.gpair, iteration);
    multi_output::reject_split_gradient(objective, iteration, &state.gpair)?;
    let per_iteration = n_out * parallel;
    for slot in 0..per_iteration {
        let k = slot / parallel;
        let gk = gather_output(&state.gpair, &mut state.gpair_k, n_out, k);
        let mut tree = std::mem::replace(
            &mut queue[iteration * per_iteration + slot],
            RegTree::with_root(0.0),
        );
        refresh_tree(&mut tree, dtrain, gk, params, refresh, tree_eta(params));
        state.margins.add_tree(&tree, TreeOutput::Scalar(k), None);
        state.model.push_tree_weighted(tree, 1.0);
    }
    Ok(())
}

/// Grow one iteration's scalar-leaf trees (gbtree or DART) with `prepared`
/// and append them.
pub(super) fn grow_round(
    run: &TrainContext,
    prepared: &Prepared,
    iteration: usize,
    state: &mut RoundState,
) -> Result<()> {
    let TrainContext {
        params,
        dtrain,
        objective,
        ..
    } = *run;
    if device_round(run, prepared, iteration, state).is_some() {
        return Ok(());
    }
    sync_host_margins(prepared, state);
    let n = dtrain.n_rows();
    let n_out = objective.n_outputs();
    let parallel = params.num_parallel_tree;
    // 1. Gradients from the current margins (all outputs at once), DART's
    //    from the ensemble minus this round's dropout set.
    let (mut rng, dropped) = round_gradients(
        run,
        &state.model,
        iteration,
        &state.margins.train,
        &mut state.gpair,
    );
    multi_output::reject_split_gradient(objective, iteration, &state.gpair)?;
    let weight = dart_new_tree_weight(dropped.as_ref(), params);
    // SGLB: the structure is searched on noisy gradients; `state.gpair`
    // keeps the noise-free ones the leaves are re-estimated from.
    let structure: &[GradPair] = match run.langevin {
        Some(langevin) => {
            langevin.structure_gradients(&state.gpair, iteration, &mut state.noisy_gpair)
        }
        None => &state.gpair,
    };

    // 2. Row subsets (uniform, class-balanced, or by query), drawn before
    //    the trees and shared across the per-output fits.
    let row_subsets = iteration_row_subsets(
        params,
        prepared.samples_per_forest(),
        run.rows,
        &state.all_rows,
        &mut rng,
    );
    // An output's gradient-based sample, when its whole forest shares one.
    let mut forest_sample = None;
    let forest_indices = prepared.forest_indices(n_out, parallel);
    let grow = GrowRound {
        run,
        prepared,
        gpair: structure,
        clean_gpair: &state.gpair,
        n_out,
        iteration,
        forest_indices: &forest_indices,
    };

    // 3. `num_parallel_tree` trees per output from the same gradients,
    //    output-major like XGBoost's layout.
    let slots: Vec<TreeSlot> = (0..n_out * parallel)
        .map(|slot| {
            let row_subset = row_subsets.rows(slot % parallel);
            // Retaining the final row partitions replaces per-row tree
            // traversals of the raw feature matrix with one sequential pass
            // per leaf: the training margin update's, when every row took
            // part, and the linear-leaf fit's. Linear leaves use them only
            // when they equal raw routing (see `rows_route_like_trees`).
            let (routed, linear_rows) = match prepared {
                Prepared::Hist {
                    rows_route_like_trees,
                    ..
                } => (true, params.linear_tree.is_some() && *rows_route_like_trees),
                // The exact builder routes rows by their raw values, as
                // prediction does.
                Prepared::Exact(_) => (true, false),
                Prepared::Approx { .. } => (false, false),
            };
            let margin_rows = routed
                && dropped.is_none()
                && row_subset.len() == n
                && !gradient_sampling(params)
                && (params.linear_tree.is_none() || linear_rows);
            TreeSlot {
                output: slot / parallel,
                parallel: slot % parallel,
                rows: row_subset,
                // Leaf re-estimation (SGLB) reads the partitions too.
                capture_rows: margin_rows || linear_rows || (routed && run.langevin.is_some()),
                margin_rows,
            }
        })
        .collect();
    // The trees of an iteration share the round's gradients and do not read
    // each other. Without gradient-based sampling or a reuse dictionary, each
    // tree's RNG draws are its column sampler and rounding seed, drawn here
    // in slot order as the sequential path draws them; the trees are then
    // grown in parallel. A GPU backend stages one tree's gradients at a
    // time, so it keeps the sequential path.
    let trees: Vec<(RegTree, Vec<LeafRows>)> = if slots.len() > 1
        && state.reuse.is_none()
        && !gradient_sampling(params)
        && params.device == Device::Cpu
        && rayon::current_num_threads() > 1
    {
        // The first tree's cuts, before any tree reads them.
        prepared.fill_approx_cache(run, gather_output(structure, &mut state.gpair_k, n_out, 0));
        let draws: Vec<(ColumnSampler, u64)> = slots
            .iter()
            .map(|_| {
                let sampler = make_column_sampler(dtrain, params, &mut rng);
                (sampler, quantization_seed(params, &mut rng))
            })
            .collect();
        // Every output's gradients gathered once, output-major, for all of
        // its parallel trees (single-output objectives read `gpair`).
        let gathered = gather_outputs(structure, n_out);
        let output_gpair = |k: usize| {
            if n_out == 1 {
                structure
            } else {
                &gathered[k * n..(k + 1) * n]
            }
        };
        // Build the forests' shared `approx` indices here, before the
        // parallel trees read them: an index built inside a tree task would
        // run its own parallel loops while other tasks wait on it, and a
        // worker waiting there can steal a task that waits on the same
        // index again.
        for (k, index) in forest_indices.iter().enumerate() {
            index.get_or_init(|| approx_index(params, dtrain, output_gpair(k), false));
        }
        slots
            .par_iter()
            .zip(draws)
            .map(|(slot, (mut sampler, rounding_seed))| {
                let gk = output_gpair(slot.output);
                let sample = TreeSample {
                    gpair: gk,
                    rows: slot.rows,
                    forest_index: grow.forest_index(slot.output),
                };
                grow_sampled_tree(&grow, slot, sample, &mut sampler, rounding_seed, None)
            })
            .collect()
    } else {
        slots
            .iter()
            .map(|slot| {
                fit_output_tree(
                    &grow,
                    slot,
                    &mut state.gpair_k,
                    &mut rng,
                    &mut forest_sample,
                    state.reuse.as_mut(),
                )
            })
            .collect::<Result<_>>()?
    };

    for (slot, (tree, leaf_rows)) in slots.iter().zip(trees) {
        // A dropout round's gradients come from the ensemble, not the margin
        // caches, which `finish_dart` recomputes.
        if dropped.is_none() {
            // The builder's final row partitions already identify the training
            // leaves when every row took part in growing the tree.
            let captured = slot.margin_rows.then_some(leaf_rows.as_slice());
            state
                .margins
                .add_tree(&tree, TreeOutput::Scalar(slot.output), captured);
        }
        state.model.push_tree_weighted(tree, weight);
    }
    if let Some(dropped) = &dropped {
        finish_dart(&mut state.model, params, dropped, &mut state.margins);
    }
    Ok(())
}

/// What the boosting rounds do to the ensemble.
pub(super) enum RoundPlan {
    /// Grow new trees with the prepared builder state.
    Grow(Prepared),
    /// `process_type=update`: refresh the queued trees of the initial model,
    /// one iteration per round, with the refresh updater's options.
    Refresh(Vec<RegTree>, Refresh),
}

/// The learning rate applied to each new tree: `eta / num_parallel_tree`
/// (XGBoost divides the rate across a forest so a whole iteration moves by
/// `eta`), in `f32` as XGBoost's `learning_rate` is.
pub(super) fn tree_eta(params: &TrainingParams) -> f32 {
    params.eta as f32 / params.num_parallel_tree as f32
}

/// Borrow the gradient slice for output `k`: the whole buffer for
/// single-output objectives, otherwise gather output `k`'s pairs into
/// `scratch` (length `n`) and borrow that.
pub(super) fn gather_output<'a>(
    gpair: &'a [GradPair],
    scratch: &'a mut [GradPair],
    n_out: usize,
    k: usize,
) -> &'a [GradPair] {
    if n_out == 1 {
        gpair
    } else {
        for (r, dst) in scratch.iter_mut().enumerate() {
            *dst = gpair[r * n_out + k];
        }
        scratch
    }
}

/// Every output's gradients of `gpair` (`[row][n_out]`) gathered
/// output-major (`[output][row]`, as [`gather_output`] gathers one); empty
/// for a single output.
fn gather_outputs(gpair: &[GradPair], n_out: usize) -> Vec<GradPair> {
    if n_out == 1 {
        return Vec::new();
    }
    let n = gpair.len() / n_out;
    let mut out = vec![GradPair::default(); gpair.len()];
    out.par_chunks_exact_mut(n)
        .enumerate()
        .for_each(|(k, column)| {
            for (r, dst) in column.iter_mut().enumerate() {
                *dst = gpair[r * n_out + k];
            }
        });
    out
}

/// One tree-growing boosting iteration: what each of its trees reads.
struct GrowRound<'a> {
    run: &'a TrainContext<'a>,
    prepared: &'a Prepared,
    /// Every output's gradients the structures are searched on,
    /// `[row][n_out]` (with Langevin noise under SGLB).
    gpair: &'a [GradPair],
    /// The noise-free gradients SGLB re-estimates the leaves from (the
    /// same as `gpair` otherwise).
    clean_gpair: &'a [GradPair],
    n_out: usize,
    /// The model's absolute iteration index.
    iteration: usize,
    /// One gradient index per output, shared by that output's forest
    /// ([`Prepared::forest_indices`]); empty when trees build their own.
    forest_indices: &'a [OnceLock<GHistIndex>],
}

impl GrowRound<'_> {
    /// The gradient index output `output`'s forest shares, if any.
    fn forest_index(&self, output: usize) -> Option<&OnceLock<GHistIndex>> {
        self.forest_indices.get(output)
    }
}

/// Which tree of an iteration to grow: parallel tree `parallel` of output
/// `output`, on the uniform row subset `rows`, keeping its leaves' rows when
/// `capture_rows` (see [`Prepared::build_tree`]) and adding it to the
/// training margins from them when `margin_rows`.
struct TreeSlot<'a> {
    output: usize,
    parallel: usize,
    rows: &'a [u32],
    capture_rows: bool,
    margin_rows: bool,
}

/// Fit the tree `slot` of the iteration `grow`: gather that output's
/// gradient slice (into `scratch` for multi-output objectives), apply
/// gradient-based row sampling when configured (per tree, as XGBoost's hist
/// updater does, or once per output forest under `approx`, kept in
/// `forest_sample` by the forest's first tree for the rest), derive its
/// column sampler, build the tree, fit linear leaves when configured, and
/// shrink its leaves by `eta / num_parallel_tree`. The caller owns the round
/// RNG (already seeded and salted), the reuse dictionary, and what happens
/// to the tree (margin updates, contribution weight).
fn fit_output_tree(
    grow: &GrowRound,
    slot: &TreeSlot,
    scratch: &mut [GradPair],
    rng: &mut Rng,
    forest_sample: &mut Option<GradientSample>,
    reuse: Option<&mut ReuseSet>,
) -> Result<(RegTree, Vec<LeafRows>)> {
    let TrainContext { params, dtrain, .. } = *grow.run;
    let (prepared, n_out) = (grow.prepared, grow.n_out);
    let gk: &[GradPair] = gather_output(grow.gpair, scratch, n_out, slot.output);
    let own;
    let sampled = if !gradient_sampling(params) {
        None
    } else if prepared.samples_per_forest() {
        if slot.parallel == 0 {
            *forest_sample = gradient_based_sample(gk, 1, params.subsample, rng)?;
        }
        forest_sample.as_ref()
    } else {
        own = gradient_based_sample(gk, 1, params.subsample, rng)?;
        own.as_ref()
    };
    let (gk, rows) = match sampled {
        Some(s) => (s.gpair.as_slice(), s.rows.as_slice()),
        None => (gk, slot.rows),
    };
    let mut sampler = make_column_sampler(dtrain, params, rng);
    let rounding_seed = quantization_seed(params, rng);
    let sample = TreeSample {
        gpair: gk,
        rows,
        forest_index: grow.forest_index(slot.output),
    };
    Ok(grow_sampled_tree(
        grow,
        slot,
        sample,
        &mut sampler,
        rounding_seed,
        reuse,
    ))
}

/// The part of [`fit_output_tree`] after its RNG draws: build the tree on
/// `sample`, fit linear leaves when configured, and shrink its leaves.
fn grow_sampled_tree(
    grow: &GrowRound,
    slot: &TreeSlot,
    sample: TreeSample,
    sampler: &mut ColumnSampler,
    rounding_seed: u64,
    reuse: Option<&mut ReuseSet>,
) -> (RegTree, Vec<LeafRows>) {
    let TrainContext { params, dtrain, .. } = *grow.run;
    let TreeSample {
        gpair: gk, rows, ..
    } = sample;
    let (mut tree, leaf_rows) = grow.prepared.build_tree(
        grow.run,
        sample,
        sampler,
        reuse,
        rounding_seed,
        slot.capture_rows,
    );
    if let Some(langevin) = grow.run.langevin {
        let at = LeafRenewal {
            data: dtrain,
            gpair: grow.clean_gpair,
            n_out: grow.n_out,
            rows,
            leaf_rows: &leaf_rows,
            iteration: grow.iteration,
            tree: slot.output * params.num_parallel_tree + slot.parallel,
        };
        langevin.renew_leaves(&mut tree, TreeOutput::Scalar(slot.output), &at);
    }
    // LightGBM keeps the first iteration's trees constant.
    if let Some(linear_tree) = params.linear_tree
        && grow.iteration > 0
    {
        let lambda = linear_tree.lambda();
        if leaf_rows.is_empty() {
            crate::tree::linear_fit::fit_linear_leaves(&mut tree, dtrain, gk, rows, lambda);
        } else {
            crate::tree::linear_fit::fit_captured_linear_leaves(
                &mut tree, dtrain, gk, &leaf_rows, lambda,
            );
        }
    }
    tree.scale_leaves(tree_eta(params));
    (tree, leaf_rows)
}

/// The stochastic-rounding seed of one quantized tree, drawn from the
/// iteration's RNG after the tree's column sampler, so every tree of an
/// iteration (outputs and parallel trees alike) rounds independently and
/// continued training resumes the same streams. Draws nothing unless
/// `use_quantized_grad` is on, leaving the default RNG streams untouched.
fn quantization_seed(params: &TrainingParams, rng: &mut Rng) -> u64 {
    if params.quantized.is_some() {
        rng.next_u64()
    } else {
        0
    }
}
