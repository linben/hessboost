//! Column (feature) subsampling shared by the tree builders.
//!
//! XGBoost exposes three cumulative column-sampling ratios, applied like its
//! `common::ColumnSampler`:
//!
//! 1. `colsample_bytree`: one pool per tree, drawn from every feature when the
//!    sampler is built;
//! 2. `colsample_bylevel`: one subset of the tree pool per depth, drawn the
//!    first time that depth is requested and cached for the rest of the tree;
//! 3. `colsample_bynode`: a fresh subset of the level set on every call.
//!
//! Each stage keeps `max(1, trunc(ratio * pool_len))` features (computed in
//! `f32`, as upstream does); a ratio of `1.0` returns its input without
//! drawing. Without feature weights a stage is a uniform draw without
//! replacement. With [`DMatrix::with_feature_weights`] it is the
//! Efraimidis–Spirakis weighted draw without replacement: every candidate
//! feature gets the key `ln(u) / max(w, 1e-6)` with `u ~ U[0, 1)`, and the
//! largest keys win. Weights below `1e-6`, zero included, are floored at
//! `1e-6` exactly as in XGBoost (`kRtEps`), so a zero weight is an epsilon
//! weight, not an exclusion: against a weight `w` a zero-weight feature still
//! wins a single draw with probability about `1e-6 / (w + 1e-6)`, and it is
//! exactly as likely as any other feature weighted below `1e-6`.
//!
//! Call granularity differs by builder: the histogram builder calls
//! [`ColumnSampler::sample`] once per node, while the exact builder (as
//! XGBoost's `colmaker` does) and symmetric (`grow_policy = symmetric`) growth
//! call it once per level, shared across that level's nodes. Interaction
//! constraints filter the returned subset afterwards and never re-add
//! unsampled features. With the default ratios of `1.0` every draw returns all
//! features.
//!
//! [`DMatrix::with_feature_weights`]: crate::data::DMatrix::with_feature_weights

use std::ops::Deref;
use std::sync::Arc;

use crate::K_RT_EPS_F32;
use crate::rng::Rng;

/// Draws feature subsets for one tree according to the `bytree`, `bylevel`,
/// and `bynode` ratios, optionally weighted by per-feature weights.
#[derive(Debug, Clone)]
pub struct ColumnSampler {
    tree: Arc<[u32]>,
    /// `bylevel` subsets by depth, drawn lazily (the tree pool itself when
    /// `bylevel` draws nothing).
    levels: Vec<Option<Arc<[u32]>>>,
    /// Per-feature sampling weights indexed by feature id; `None` samples
    /// uniformly.
    weights: Option<Vec<f32>>,
    bylevel: f32,
    bynode: f32,
    rng: Rng,
    /// The seed `rng` started from: the tree's own seed, which the trainer
    /// derives from the configured seed, round, and output.
    seed: u64,
}

/// The candidate features [`ColumnSampler::sample`] returns, ascending: a
/// node's own `bynode` draw, or the tree's (or level's) set shared without a
/// copy when the node stage draws nothing.
#[derive(Debug, Clone)]
pub enum FeatureSet {
    /// A set the sampler keeps: the tree's pool or a depth's `bylevel` set.
    Shared(Arc<[u32]>),
    /// A node's `bynode` draw.
    Drawn(Vec<u32>),
}

impl Deref for FeatureSet {
    type Target = [u32];

    #[inline]
    fn deref(&self) -> &[u32] {
        match self {
            FeatureSet::Shared(features) => features,
            FeatureSet::Drawn(features) => features,
        }
    }
}

impl PartialEq for FeatureSet {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for FeatureSet {}

impl ColumnSampler {
    /// Build the sampler for one tree over `n_features` columns, drawing the
    /// `bytree` pool immediately. `weights`, when given, has one non-negative
    /// entry per feature and makes every stage a weighted draw.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is given with a length other than `n_features`.
    pub fn new(
        n_features: usize,
        weights: Option<&[f32]>,
        bytree: f64,
        bylevel: f64,
        bynode: f64,
        seed: u64,
    ) -> Self {
        if let Some(w) = weights {
            assert_eq!(w.len(), n_features, "one feature weight per feature");
        }
        let weights = weights.map(<[f32]>::to_vec);
        let mut rng = Rng::new(seed);
        let all: Vec<u32> = (0..n_features as u32).collect();
        let tree = draw(&mut rng, weights.as_deref(), &all, bytree as f32).unwrap_or(all);
        ColumnSampler {
            tree: tree.into(),
            levels: Vec::new(),
            weights,
            bylevel: bylevel as f32,
            bynode: bynode as f32,
            rng,
            seed,
        }
    }

    /// A pass-through sampler over all `n_features` columns (ratios `1.0`),
    /// primarily for tests and callers that do no column sampling.
    pub fn all(n_features: usize) -> Self {
        ColumnSampler::new(n_features, None, 1.0, 1.0, 1.0, 0)
    }

    /// A sampler that offers exactly `features` (ascending) at every node,
    /// drawing nothing: an EBM term's tree. `seed` keys the tree's other
    /// random streams ([`Self::seed`]).
    pub(crate) fn only(features: Vec<u32>, seed: u64) -> Self {
        ColumnSampler {
            tree: features.into(),
            levels: Vec::new(),
            weights: None,
            bylevel: 1.0,
            bynode: 1.0,
            rng: Rng::new(seed),
            seed,
        }
    }

    /// The candidate features for a node at `depth`: that depth's cached
    /// `bylevel` subset of the tree pool, then a fresh `bynode` subset of it.
    /// Returned features are sorted ascending.
    pub fn sample(&mut self, depth: usize) -> FeatureSet {
        if let Some(features) = self.fixed_features() {
            return features;
        }
        if self.levels.len() <= depth {
            self.levels.resize(depth + 1, None);
        }
        let weights = self.weights.as_deref();
        let tree = &self.tree;
        let level = self.levels[depth].get_or_insert_with(|| {
            draw(&mut self.rng, weights, tree, self.bylevel)
                .map_or_else(|| Arc::clone(tree), Arc::from)
        });
        match draw(&mut self.rng, weights, level, self.bynode) {
            Some(drawn) => FeatureSet::Drawn(drawn),
            None => FeatureSet::Shared(Arc::clone(level)),
        }
    }

    /// What [`Self::sample`] returns at every depth when it draws nothing
    /// (no `bylevel` or `bynode` sampling), else `None`.
    pub(crate) fn fixed_features(&self) -> Option<FeatureSet> {
        (self.bylevel >= 1.0 && self.bynode >= 1.0)
            .then(|| FeatureSet::Shared(Arc::clone(&self.tree)))
    }

    /// The per-tree seed this sampler was built with. Other per-tree random
    /// streams (the `extra_trees` threshold draws) derive from it so they vary
    /// across rounds and outputs without consuming this sampler's draws.
    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }
}

/// One sampling stage over `pool`: `max(1, trunc(ratio * len))` features
/// without replacement (weighted when `weights` is set), sorted ascending.
/// `None` when the stage keeps the whole pool (`ratio >= 1` or an empty
/// pool), which draws nothing.
fn draw(rng: &mut Rng, weights: Option<&[f32]>, pool: &[u32], ratio: f32) -> Option<Vec<u32>> {
    if ratio >= 1.0 || pool.is_empty() {
        return None;
    }
    let n = ((ratio * pool.len() as f32) as usize).clamp(1, pool.len());
    let mut chosen = match weights {
        None => {
            let mut features = pool.to_vec();
            rng.shuffle(&mut features);
            features.truncate(n);
            features
        }
        Some(weights) => {
            // XGBoost floors every weight at `kRtEps` before the draw.
            let mut keyed: Vec<(f32, u32)> = pool
                .iter()
                .map(|&f| {
                    let w = weights[f as usize].max(K_RT_EPS_F32);
                    (rng.f32().ln() / w, f)
                })
                .collect();
            // Stable descending sort by key, as XGBoost's `ArgSort`.
            keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
            keyed.truncate(n);
            keyed.into_iter().map(|(_, f)| f).collect()
        }
    };
    chosen.sort_unstable();
    Some(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_through_when_ratios_one() {
        let mut s = ColumnSampler::all(10);
        let all: Vec<u32> = (0..10).collect();
        assert_eq!(*s.sample(0), all);
        assert_eq!(*s.sample(3), all);
    }

    #[test]
    fn stage_counts_truncate_like_xgboost() {
        // 10 * 0.75 = 7.5 -> 7 (bytree); 7 * 0.5 = 3.5 -> 3 (bylevel);
        // 3 * 0.5 = 1.5 -> 1 (bynode).
        let mut s = ColumnSampler::new(10, None, 0.75, 0.5, 0.5, 42);
        assert_eq!(s.tree.len(), 7);
        let f = s.sample(0);
        assert_eq!(f.len(), 1);
        assert_eq!(s.levels[0].as_ref().unwrap().len(), 3);
        // A tiny ratio still keeps one feature.
        let mut tiny = ColumnSampler::new(3, None, 0.01, 0.01, 0.01, 1);
        assert_eq!(tiny.sample(0).len(), 1);
    }

    #[test]
    fn level_subset_is_cached_per_depth_and_node_subsets_nest_in_it() {
        let mut s = ColumnSampler::new(40, None, 0.8, 0.5, 0.5, 9);
        let tree = s.tree.clone();
        let first = s.sample(2);
        let level = s.levels[2].clone().unwrap();
        assert!(level.iter().all(|f| tree.contains(f)));
        let mut node_sets = vec![first];
        for _ in 0..20 {
            node_sets.push(s.sample(2));
        }
        assert_eq!(s.levels[2].as_ref().unwrap(), &level, "level set is fixed");
        for set in &node_sets {
            assert!(set.windows(2).all(|w| w[0] < w[1]), "sorted & unique");
            assert!(set.iter().all(|f| level.contains(f)));
        }
        assert!(
            node_sets.iter().any(|set| set != &node_sets[0]),
            "node subsets are redrawn per call"
        );
    }

    #[test]
    fn deterministic_for_seed() {
        let weights: Vec<f32> = (0..50).map(|i| (i % 7) as f32).collect();
        for w in [None, Some(weights.as_slice())] {
            let mut a = ColumnSampler::new(50, w, 0.6, 0.7, 0.8, 7);
            let mut b = ColumnSampler::new(50, w, 0.6, 0.7, 0.8, 7);
            for depth in [0, 1, 1, 0, 3] {
                assert_eq!(a.sample(depth), b.sample(depth));
            }
        }
    }

    /// Selecting one feature out of a pool with the ES keys picks feature `i`
    /// with probability `w_i / sum(w)`; check the empirical frequencies.
    #[test]
    fn single_draw_frequencies_are_proportional_to_weights() {
        let weights = [1.0f32, 2.0, 3.0, 4.0, 0.0];
        let trials = 40_000;
        let mut counts = [0usize; 5];
        for seed in 0..trials {
            // 5 * 0.2 = 1 feature per tree.
            let s = ColumnSampler::new(5, Some(&weights), 0.2, 1.0, 1.0, seed);
            counts[s.tree[0] as usize] += 1;
        }
        // The zero weight is an epsilon weight (win chance ~1e-7 per draw
        // here), not an exclusion; these seeds never draw it.
        assert_eq!(counts[4], 0, "an epsilon weight almost never wins");
        for (i, &count) in counts[..4].iter().enumerate() {
            let expected = f64::from(weights[i]) / 10.0;
            let freq = count as f64 / f64::from(trials as u32);
            // Binomial standard error is below 0.0025; allow ~5 sigma.
            assert!(
                (freq - expected).abs() < 0.012,
                "feature {i}: freq {freq} vs {expected}"
            );
        }
    }

    /// Multi-feature draws without replacement: inclusion probabilities follow
    /// the successive-sampling law, which for two of three features with
    /// weights (1, 1, 8) gives feature 2 probability
    /// 0.8 + 2 * 0.1 * (8 / 9) = 0.9778 and each light feature 0.5111.
    #[test]
    fn without_replacement_inclusion_matches_successive_sampling() {
        let weights = [1.0f32, 1.0, 8.0];
        let trials = 30_000u64;
        let mut counts = [0usize; 3];
        for seed in 0..trials {
            // 3 * 0.7 = 2.1 -> 2 features at the node stage.
            let mut s = ColumnSampler::new(3, Some(&weights), 1.0, 1.0, 0.7, seed);
            let f = s.sample(0);
            assert_eq!(f.len(), 2);
            for &x in f.iter() {
                counts[x as usize] += 1;
            }
        }
        let freq = |i: usize| counts[i] as f64 / trials as f64;
        let heavy = 0.8 + 2.0 * 0.1 * (8.0 / 9.0);
        let light = (2.0 - heavy) / 2.0;
        assert!((freq(2) - heavy).abs() < 0.01, "heavy {}", freq(2));
        assert!((freq(0) - light).abs() < 0.015, "light0 {}", freq(0));
        assert!((freq(1) - light).abs() < 0.015, "light1 {}", freq(1));
    }

    /// Zero weights are floored to the `1e-6` epsilon: against weights far
    /// above it they practically never win (fixed seeds here), so a stage
    /// larger than the positive-weight count fills from them.
    #[test]
    fn zero_weight_features_rarely_beat_large_weights() {
        // Two positive features, a stage of three: the third comes from the
        // zero-weight features.
        let weights = [0.0f32, 5.0, 0.0, 1.0, 0.0];
        for seed in 0..200 {
            let mut s = ColumnSampler::new(5, Some(&weights), 1.0, 0.4, 1.0, seed);
            let two = s.sample(0);
            assert_eq!(*two, [1, 3], "large weights win on these seeds");
            let mut s = ColumnSampler::new(5, Some(&weights), 0.6, 1.0, 1.0, seed);
            let three = s.sample(0);
            assert_eq!(three.len(), 3);
            assert!(three.contains(&1) && three.contains(&3));
        }
    }

    /// A zero weight is not an exclusion: it ties with a positive weight below
    /// the `1e-6` floor, so both are drawn equally often and with the same
    /// seeds as two zero weights.
    #[test]
    fn zero_and_below_epsilon_weights_are_drawn_alike() {
        let trials = 4_000u64;
        let mut zero_drawn = 0usize;
        for seed in 0..trials {
            let s = ColumnSampler::new(2, Some(&[0.0, 1e-8]), 0.5, 1.0, 1.0, seed);
            let floored = ColumnSampler::new(2, Some(&[0.0, 0.0]), 0.5, 1.0, 1.0, seed);
            assert_eq!(s.tree, floored.tree);
            zero_drawn += usize::from(*s.tree == [0]);
        }
        let freq = zero_drawn as f64 / trials as f64;
        // Binomial standard error is ~0.008; allow ~5 sigma.
        assert!((freq - 0.5).abs() < 0.04, "zero-weight frequency {freq}");
    }
}
