//! Builder state prepared once per run: the training context, `tree_method`'s
//! builder (`exact`, `hist` with its backend, `approx` with its cuts).

use super::dart::{round_rng, round_salt};
use super::round::gather_output;
use super::row_sampling::{RowMeta, gradient_sampling};
use crate::config::{Device, GrowPolicy, TrainingParams, TreeMethod};
use crate::data::ghist::GHistIndex;
use crate::data::quantile::HistCuts;
use crate::data::{DMatrix, MetaInfo};
use crate::error::{HessboostError, Result};
use crate::objective::{GradPair, Loss};
use crate::training::sampling::gradient_based_sample;
use crate::training::sglb::Langevin;
use crate::tree::RegTree;
use crate::tree::builder::{
    ExactTreeBuilder, HistTreeBuilder, LeafRows, SortedColumns, check_symmetric_input,
};
use crate::tree::hist::{CpuBackend, HistogramBackend};
use crate::tree::reuse::ReuseSet;
use crate::tree::sampler::ColumnSampler;
use std::sync::OnceLock;

/// What every boosting round of one training run reads: the parameters, the
/// training matrix with its metadata, the objective, and the Langevin noise
/// (`None` unless SGLB is on).
#[derive(Clone, Copy)]
pub(super) struct TrainContext<'a> {
    pub(super) params: &'a TrainingParams,
    pub(super) dtrain: &'a DMatrix,
    pub(super) info: &'a MetaInfo<'a>,
    pub(super) objective: &'a dyn Loss,
    pub(super) langevin: Option<&'a Langevin>,
    /// What row sampling reads of `dtrain`, gathered once per run.
    pub(super) rows: RowMeta<'a>,
}

/// The gradients one tree grows on and the rows that take part: an output's
/// gradients and its uniform row subset, or their gradient-based sample.
#[derive(Clone, Copy)]
pub(super) struct TreeSample<'a> {
    pub(super) gpair: &'a [GradPair],
    pub(super) rows: &'a [u32],
    /// Under `approx` with per-round cuts, the gradient index that every
    /// tree of this output's forest builds from these same gradients
    /// (`None` elsewhere): built before the parallel trees start, or by the
    /// first tree that needs it on the serial path.
    pub(super) forest_index: Option<&'a OnceLock<GHistIndex>>,
}

/// Prepared, reusable per-round builder state, chosen by `tree_method`.
pub(super) enum Prepared {
    Exact(SortedColumns),
    /// Histogram method: the binned dataset plus the backend its histograms
    /// are built on (the CPU's, or the Metal GPU's when `device = metal`).
    Hist {
        index: GHistIndex,
        backend: Box<dyn HistogramBackend>,
        /// Every training value was sketched (no row has a zero sample
        /// weight), so the builder's row partitions equal routing each
        /// training row through the finished tree by raw value: numeric
        /// values lie below their feature's last cut, and categorical cuts
        /// hold every category. A zero-weight row's value can lie beyond the
        /// last cut, where binning clamps it into the last bin while the tree
        /// routes it by its threshold, so linear leaves then route instead of
        /// reading the builder's rows.
        rows_route_like_trees: bool,
    },
    /// `tree_method=approx`: Hessian-weighted cuts. XGBoost regenerates them
    /// every round from a sorted-column summary unless the objective has a
    /// constant Hessian, in which case the first tree's streaming sketch (of
    /// its sampled Hessians) is built once and reused (`BatchParam::regen =
    /// !const_hess`); continued training replays it
    /// ([`Prepared::resume_approx_cache`]).
    Approx {
        const_hess: bool,
        cached: OnceLock<GHistIndex>,
    },
}

impl Prepared {
    /// The binned index and backend of a histogram run whose backend keeps
    /// the rows on a device (the CUDA backend), for device-resident rounds.
    pub(super) fn device_backend(&self) -> Option<(&GHistIndex, &dyn HistogramBackend)> {
        match self {
            Prepared::Hist { index, backend, .. } if backend.row_engine().is_some() => {
                Some((index, backend.as_ref()))
            }
            _ => None,
        }
    }

    /// Grow one tree on `sample`, with the rows that reached each leaf when
    /// `capture_rows` (histogram and exact methods; empty otherwise). With reuse
    /// penalties (`reuse` is `Some`) the split search is penalized by the
    /// ensemble's dictionary, which the new tree's splits then extend.
    /// `rounding_seed` keys the stochastic rounding of quantized training.
    pub(super) fn build_tree(
        &self,
        run: &TrainContext,
        sample: TreeSample,
        sampler: &mut ColumnSampler,
        reuse: Option<&mut ReuseSet>,
        rounding_seed: u64,
        capture_rows: bool,
    ) -> (RegTree, Vec<LeafRows>) {
        let TrainContext { params, dtrain, .. } = *run;
        let TreeSample {
            gpair,
            rows,
            forest_index,
        } = sample;
        let hist = |ghist: &GHistIndex,
                    backend: &dyn HistogramBackend,
                    reuse: Option<&ReuseSet>,
                    sampler: &mut ColumnSampler| {
            let builder = HistTreeBuilder::new(params)
                .with_rounding_seed(rounding_seed)
                .with_reuse(reuse, ghist.cuts())
                .with_backend(backend);
            if capture_rows {
                builder.build_with_leaf_rows(ghist, gpair, rows, sampler)
            } else {
                (builder.build(ghist, gpair, rows, sampler), Vec::new())
            }
        };
        let (tree, leaf_rows) = match self {
            Prepared::Exact(cols) => {
                let builder = ExactTreeBuilder::new(params).with_reuse(reuse.as_deref());
                if capture_rows {
                    builder.build_with_leaf_rows(cols, dtrain, gpair, rows, sampler)
                } else {
                    (
                        builder.build(cols, dtrain, gpair, rows, sampler),
                        Vec::new(),
                    )
                }
            }
            Prepared::Hist { index, backend, .. } => {
                hist(index, backend.as_ref(), reuse.as_deref(), sampler)
            }
            Prepared::Approx { const_hess, cached } => {
                let bin = || approx_index(params, dtrain, gpair, *const_hess);
                // `approx` never runs with a GPU device: `validate` refuses
                // the combination, so its histograms always use the CPU.
                let cpu = CpuBackend;
                if *const_hess {
                    hist(cached.get_or_init(bin), &cpu, reuse.as_deref(), sampler)
                } else if let Some(shared) = forest_index {
                    hist(shared.get_or_init(bin), &cpu, reuse.as_deref(), sampler)
                } else {
                    hist(&bin(), &cpu, reuse.as_deref(), sampler)
                }
            }
        };
        if let Some(reuse) = reuse {
            reuse.record_tree(&tree);
        }
        (tree, leaf_rows)
    }

    /// The per-round gradient indices of `approx` with a non-constant
    /// Hessian when a forest has several trees: one per output, built before
    /// the parallel trees start (or by the first tree that needs it on the
    /// serial path), since every tree of an output's forest
    /// weights its cuts by the same gradients (one row sample and, under
    /// gradient-based sampling, one gradient sample serve the whole forest,
    /// [`Self::samples_per_forest`]). Empty otherwise.
    pub(super) fn forest_indices(
        &self,
        n_out: usize,
        num_parallel_tree: usize,
    ) -> Vec<OnceLock<GHistIndex>> {
        match self {
            Prepared::Approx {
                const_hess: false, ..
            } if num_parallel_tree > 1 => (0..n_out).map(|_| OnceLock::new()).collect(),
            _ => Vec::new(),
        }
    }

    /// Whether one row sample serves every parallel tree of an output: XGBoost
    /// 3.4.2's `GlobalApproxUpdater::Update` samples once before its tree loop
    /// and grows the whole forest from those gradients and sketch Hessians,
    /// whereas the hist and exact updaters sample each tree.
    pub(super) fn samples_per_forest(&self) -> bool {
        matches!(self, Prepared::Approx { .. })
    }

    /// Continued training: seed the constant-Hessian `approx` cache with the
    /// cuts an uninterrupted run holds. That run cached the cuts of its first
    /// tree (iteration 0, output 0), whose Hessians gradient-based sampling
    /// zeroes or rescales, so they depend on that tree's sample; XGBoost
    /// keeps them in the training matrix's gradient-index cache, which a
    /// continuation on the same matrix reuses. Replays iteration 0 from
    /// `margin0` (the intercept margins): its gradients and its RNG draws up
    /// to the first sample. Without gradient sampling every round's constant
    /// Hessians agree, and the cache fills lazily as usual.
    pub(super) fn resume_approx_cache(
        &self,
        run: &TrainContext,
        margin0: &[f32],
        gpair: &mut [GradPair],
        gpair_k: &mut [GradPair],
        n_out: usize,
    ) -> Result<()> {
        let TrainContext {
            params,
            info,
            objective,
            ..
        } = *run;
        let const_hess_approx = matches!(
            self,
            Prepared::Approx {
                const_hess: true,
                ..
            }
        );
        if !const_hess_approx || !gradient_sampling(params) {
            return Ok(());
        }
        // Iteration 0's stream before its first sample: `select_dropout`
        // draws nothing over the empty ensemble, and `sample_rows` draws
        // nothing under gradient sampling.
        let mut rng = round_rng(params, 0, round_salt(params));
        objective.gradient_info(margin0, info, gpair);
        let g0 = gather_output(gpair, gpair_k, n_out, 0);
        let sampled = gradient_based_sample(g0, 1, params.subsample, &mut rng)?;
        let g0 = sampled.as_ref().map_or(g0, |s| s.gpair.as_slice());
        self.fill_approx_cache(run, g0);
        Ok(())
    }

    /// Build the constant-Hessian `approx` cache from `gpair`, the gradients
    /// its first tree reads, unless it is already built (a no-op for every
    /// other builder). The first tree of a run is output 0's, so the parallel
    /// slot loop calls this with output 0's gradients before growing trees
    /// for several outputs at once: whichever tree ran first would otherwise
    /// pick the cuts, and a custom objective's constant Hessians may differ
    /// by output. XGBoost 3.4.2 likewise keeps the first gradient index its
    /// training matrix builds (`BatchParam::regen` is false), whichever
    /// output group later reads it.
    pub(super) fn fill_approx_cache(&self, run: &TrainContext, gpair: &[GradPair]) {
        if let Prepared::Approx {
            const_hess: true,
            cached,
        } = self
        {
            cached.get_or_init(|| approx_index(run.params, run.dtrain, gpair, true));
        }
    }
}

/// The `approx` gradient index of one tree: cuts weighted by `gpair`'s
/// Hessians (a sorted-column summary unless the Hessian is constant).
pub(super) fn approx_index(
    params: &TrainingParams,
    dtrain: &DMatrix,
    gpair: &[GradPair],
    const_hess: bool,
) -> GHistIndex {
    let cuts =
        HistCuts::from_dmatrix_hessians(dtrain, params.max_bin, |row| gpair[row].hess, !const_hess);
    GHistIndex::from_dmatrix(dtrain, cuts)
}

/// Resolve `tree_method` (handling `Auto`) and prepare the matching builder
/// state once, up front. `const_hess` is the objective's
/// [`Loss::const_hess`](crate::objective::Loss::const_hess), which
/// decides whether `approx` regenerates its cuts every round.
pub(super) fn prepare_builder(
    params: &TrainingParams,
    dtrain: &DMatrix,
    const_hess: bool,
) -> Result<Prepared> {
    let method = match params.tree_method {
        // Auto favors the histogram method, as modern XGBoost does.
        TreeMethod::Auto | TreeMethod::Hist => TreeMethod::Hist,
        TreeMethod::Exact => TreeMethod::Exact,
        TreeMethod::Approx => TreeMethod::Approx,
    };
    if method == TreeMethod::Exact && gradient_sampling(params) {
        return Err(HessboostError::invalid_param(
            "sampling_method",
            "`gradient_based` sampling requires `tree_method=hist` or `approx`; \
             `exact` supports only `uniform`",
        ));
    }
    if method == TreeMethod::Exact && params.grow_policy == GrowPolicy::LossGuide {
        return Err(HessboostError::invalid_param(
            "grow_policy",
            "`lossguide` growth requires `tree_method=hist`",
        ));
    }
    if params.grow_policy == GrowPolicy::Symmetric {
        check_symmetric_input(method, dtrain)?;
    }
    Ok(match method {
        TreeMethod::Hist => {
            let cuts = HistCuts::from_dmatrix(dtrain, params.max_bin);
            let index = GHistIndex::from_dmatrix(dtrain, cuts);
            let backend = hist_backend(params, &index)?;
            let rows_route_like_trees = dtrain.weights().is_none_or(|w| !w.contains(&0.0));
            Prepared::Hist {
                index,
                backend,
                rows_route_like_trees,
            }
        }
        TreeMethod::Approx => Prepared::Approx {
            const_hess,
            cached: OnceLock::new(),
        },
        _ => Prepared::Exact(SortedColumns::from_dmatrix(dtrain)),
    })
}

/// The histogram backend a training run builds on: the Metal GPU's when
/// `device = metal`, the CUDA GPU's when `device = cuda` (the parameter
/// validation has already checked the platform and feature), else the
/// CPU's.
fn hist_backend(params: &TrainingParams, index: &GHistIndex) -> Result<Box<dyn HistogramBackend>> {
    match params.device {
        Device::Cpu => {
            let backend: Box<dyn HistogramBackend> = Box::new(CpuBackend);
            Ok(backend)
        }
        Device::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                let backend: Box<dyn HistogramBackend> =
                    Box::new(crate::backend::metal::MetalHistBackend::new(index)?);
                Ok(backend)
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                // Unreachable in practice: `TrainingParams::validate` refuses
                // `device = metal` without the feature, and training always
                // validates first.
                let _ = index;
                Err(HessboostError::invalid_param(
                    "device",
                    "`metal` requires building with the `metal` feature on macOS",
                ))
            }
        }
        Device::Cuda { ordinal } => {
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            {
                let backend: Box<dyn HistogramBackend> =
                    Box::new(crate::backend::cuda::CudaHistBackend::new(index, ordinal)?);
                Ok(backend)
            }
            #[cfg(not(all(target_os = "linux", feature = "cuda")))]
            {
                // Unreachable in practice, as for Metal above.
                let _ = (index, ordinal);
                Err(HessboostError::invalid_param(
                    "device",
                    "`cuda` requires building with the `cuda` feature on Linux",
                ))
            }
        }
    }
}
