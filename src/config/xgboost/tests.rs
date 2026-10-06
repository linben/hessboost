use crate::config::*;
use crate::error::HessboostError;
use crate::metric::EvalMetric;
use crate::objective::distributional::{DistFamily, Distributional};
use crate::objective::distributional::{DistGradient, DistSplitDirection};
use crate::objective::{
    Aft, Expectiles, LambdaRank, Multiclass, PseudoHuber, Quantiles, RegLoss, Tweedie,
};
use crate::objective::{AftDistribution, Objective};
use serde_json::json;
use serde_json::{Map, Value};
use std::num::NonZeroUsize;

/// The parameter [`TrainingParams::from_xgboost`] refuses, if any.
fn refused(pairs: Value) -> Option<String> {
    let Value::Object(map) = pairs else {
        unreachable!("test input is an object")
    };
    TrainingParams::from_xgboost(map)
        .err()
        .map(|e| e.to_string())
}

/// XGBoost's key and value spellings configure the same settings, and a
/// configuration's flat form reads back to itself.
#[test]
fn xgboost_spellings_parse_and_round_trip() {
    let p = TrainingParams::from_xgboost([
        ("objective", json!("reg:quantileerror")),
        ("aft_loss_distribution", json!("extreme")),
        ("sampling_method", json!("gradient_based")),
        ("multi_strategy", json!("multi_output_tree")),
        ("process_type", json!("update")),
        ("refresh_leaf", json!(false)),
        ("num_parallel_tree", json!(4)),
        ("quantile_alpha", json!(0.25)),
        ("eval_metric", json!("aft-nloglik")),
        ("monotone_constraints", json!("(1,-1,0)")),
        ("interaction_constraints", json!("[[0, 1], [2]]")),
        ("learning_rate", json!(0.1)),
        ("reg_lambda", json!(2.0)),
        ("max_delta_step", json!(0.0)),
    ])
    .unwrap();
    assert_eq!(
        p.objective,
        Objective::Quantile(Quantiles::new([0.25]).unwrap())
    );
    assert_eq!(
        p.eval_metric,
        [EvalMetric::AftNLogLik(Aft::with_distribution(
            AftDistribution::Extreme
        ))]
    );
    assert_eq!(p.sampling_method, SamplingMethod::GradientBased);
    assert_eq!(p.multi_strategy, MultiStrategy::MultiOutputTree);
    assert_eq!(p.process_type, ProcessType::Update(Refresh::stats_only()));
    assert_eq!(p.num_parallel_tree, 4);
    assert_eq!(
        p.monotone_constraints,
        [Monotone::Increasing, Monotone::Decreasing, Monotone::None]
    );
    assert_eq!(p.interaction_constraints, [vec![0, 1], vec![2]]);
    assert_eq!((p.eta, p.lambda), (0.1, 2.0));
    assert_eq!(p.max_delta_step, MaxDeltaStep::Unbounded);

    let flat = p.to_xgboost().unwrap();
    let back = TrainingParams::from_xgboost(flat.clone()).unwrap();
    assert_eq!(back.to_xgboost().unwrap(), flat);
    let defaults = TrainingParams::from_xgboost(Map::new()).unwrap();
    assert_eq!(
        defaults.to_xgboost().unwrap(),
        TrainingParams::default().to_xgboost().unwrap()
    );
    assert!(
        !defaults
            .to_xgboost()
            .unwrap()
            .contains_key("max_delta_step")
    );
    let rank = TrainingParams::from_xgboost([
        ("objective", json!("rank:ndcg")),
        ("lambdarank_pair_method", json!("topk")),
    ])
    .unwrap();
    assert_eq!(rank.objective, Objective::RankNdcg(LambdaRank::default()));
}

/// XGBoost's `device` spellings: `cuda` and its alias `gpu`, each with an
/// optional ordinal; a device serializes back to `cpu`, `metal`, `cuda`, or
/// `cuda:<ordinal>`. Anything else is refused rather than read as a device.
#[test]
fn device_spellings_parse_and_round_trip() {
    let parse = |s: &str| serde_json::from_value::<Device>(json!(s)).ok();
    let cuda = |ordinal| Some(Device::Cuda { ordinal });
    assert_eq!(parse("cpu"), Some(Device::Cpu));
    assert_eq!(parse("metal"), Some(Device::Metal));
    assert_eq!(parse("cuda"), cuda(0));
    assert_eq!(parse("gpu"), cuda(0));
    assert_eq!(parse("cuda:3"), cuda(3));
    assert_eq!(parse("gpu:12"), cuda(12));
    for bad in [
        "", "CUDA", "cuda:", "cuda:x", "cuda:+1", "gpu:-1", "cuda:1:2", "cuda: 1",
    ] {
        assert_eq!(parse(bad), None, "{bad:?}");
    }
    assert_eq!(json!(Device::Cuda { ordinal: 0 }), json!("cuda"));
    for device in [
        Device::Cpu,
        Device::Metal,
        Device::Cuda { ordinal: 0 },
        Device::Cuda { ordinal: 7 },
    ] {
        assert_eq!(parse(json!(device).as_str().unwrap()), Some(device));
    }
    // Through the flat form: accepted where the backend is compiled in,
    // otherwise refused under `device`.
    let flat = TrainingParams::from_xgboost([("device", json!("gpu:2"))]);
    if cfg!(all(target_os = "linux", feature = "cuda")) {
        assert_eq!(flat.unwrap().device, Device::Cuda { ordinal: 2 });
    } else {
        assert!(matches!(
            flat,
            Err(HessboostError::InvalidParameter { name: "device", .. })
        ));
    }
}

/// A key of an option group means nothing while the group's switch is
/// off, so it is refused by name rather than ignored.
#[test]
fn dependent_keys_without_their_switch_are_refused_by_name() {
    for (pairs, key) in [
        (json!({"booster": "gbtree", "rate_drop": 0.1}), "rate_drop"),
        (json!({"one_drop": true}), "one_drop"),
        (json!({"refresh_leaf": false}), "refresh_leaf"),
        (json!({"extra_seed": 3}), "extra_seed"),
        (json!({"extra_trees": false, "extra_seed": 3}), "extra_seed"),
        (json!({"linear_lambda": 0.5}), "linear_lambda"),
        (json!({"num_grad_quant_bins": 8}), "num_grad_quant_bins"),
    ] {
        let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
        assert!(
            refusal.starts_with(&format!("invalid parameter `{key}`")),
            "{pairs}: {refusal}"
        );
    }
}

/// LightGBM's class fractions become one [`BalancedBagging`] (a missing
/// one at LightGBM's default 1, both at 1 meaning off) and read back
/// from the flat form; the fractions are refused by name where nothing
/// would bag by class.
#[test]
fn balanced_bagging_reads_lightgbm_fractions() {
    let binary = json!("binary:logistic");
    let p = TrainingParams::from_xgboost([
        ("objective", binary.clone()),
        ("neg_bagging_fraction", json!(0.2)),
    ])
    .unwrap();
    assert_eq!(
        p.balanced_bagging,
        Some(BalancedBagging::new(1.0, 0.2).unwrap())
    );
    assert_eq!(
        TrainingParams::from_xgboost(p.to_xgboost().unwrap()).unwrap(),
        p
    );
    let off = TrainingParams::from_xgboost([
        ("objective", binary.clone()),
        ("pos_bagging_fraction", json!(1.0)),
        ("neg_bagging_fraction", json!(1.0)),
    ])
    .unwrap();
    assert_eq!(off.balanced_bagging, None);
    for (pairs, key) in [
        (json!({"pos_bagging_fraction": 0.5}), "pos_bagging_fraction"),
        (
            json!({"objective": binary, "neg_bagging_fraction": 0.5, "subsample": 0.8}),
            "subsample",
        ),
        (
            json!({"objective": binary, "neg_bagging_fraction": 0.0}),
            "neg_bagging_fraction",
        ),
    ] {
        let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
        assert!(
            refusal.starts_with(&format!("invalid parameter `{key}`")),
            "{pairs}: {refusal}"
        );
    }
}

/// LightGBM's `bagging_by_query` reads `subsample` as the fraction of
/// queries kept, and writes it back there; without a fraction below 1,
/// or with a non-ranking objective, it is refused by name.
#[test]
fn bagging_by_query_reads_subsample_as_the_query_fraction() {
    let p = TrainingParams::from_xgboost([
        ("objective", json!("rank:xendcg")),
        ("bagging_by_query", json!(true)),
        ("subsample", json!(0.7)),
    ])
    .unwrap();
    assert_eq!(p.bagging_by_query, Some(QueryBagging::new(0.7).unwrap()));
    assert_eq!(p.subsample, 1.0);
    let flat = p.to_xgboost().unwrap();
    assert_eq!(
        (&flat["bagging_by_query"], &flat["subsample"]),
        (&json!(true), &json!(0.7))
    );
    assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), p);
    let off = TrainingParams::from_xgboost([
        ("bagging_by_query", json!(false)),
        ("subsample", json!(0.7)),
    ])
    .unwrap();
    assert_eq!((off.bagging_by_query, off.subsample), (None, 0.7));
    for pairs in [
        json!({"objective": "rank:ndcg", "bagging_by_query": true}),
        json!({"bagging_by_query": true, "subsample": 0.5}),
    ] {
        let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
        assert!(
            refusal.starts_with("invalid parameter `bagging_by_query`"),
            "{pairs}: {refusal}"
        );
    }
}

/// Every option group, switched on with non-default values, reads back
/// from its flat form unchanged.
#[test]
fn option_groups_round_trip_through_the_flat_form() {
    let p = TrainingParams {
        booster: BoosterKind::Dart(
            Dart::builder()
                .rate_drop(0.2)
                .skip_drop(0.3)
                .one_drop(true)
                .build()
                .unwrap(),
        ),
        process_type: ProcessType::Update(Refresh::stats_only()),
        extra_trees: Some(ExtraTrees::with_seed(11)),
        linear_tree: Some(LinearTree::new(0.5).unwrap()),
        quantized: Some(
            QuantizedGrad::builder()
                .bins(8)
                .stochastic_rounding(false)
                .renew_leaf(true)
                .build()
                .unwrap(),
        ),
        ..TrainingParams::default()
    };
    let flat = p.to_xgboost().unwrap();
    assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), p);
}

/// XGBoost's `0` limits are `None`, and `max_delta_step` keeps its three
/// states: absent or `null` is the objective's default, `0` no bound,
/// anything else a bound. Each reads back from its flat form.
#[test]
fn sentinels_map_to_typed_states_and_back() {
    let parse = |pairs: Value| TrainingParams::from_xgboost(pairs.as_object().unwrap().clone());
    let p =
        parse(json!({"max_depth": 0, "max_leaves": 0, "nthread": 0, "grow_policy": "depthwise"}))
            .unwrap();
    assert_eq!((p.max_depth, p.max_leaves, p.nthread), (None, None, None));
    let p = parse(json!({"max_depth": 3, "max_leaves": 7, "nthread": 2})).unwrap();
    assert_eq!(
        (p.max_depth, p.max_leaves, p.nthread),
        (
            NonZeroUsize::new(3),
            NonZeroUsize::new(7),
            NonZeroUsize::new(2)
        )
    );
    for (value, step) in [
        (None, MaxDeltaStep::ObjectiveDefault),
        (Some(json!(null)), MaxDeltaStep::ObjectiveDefault),
        (Some(json!(0.0)), MaxDeltaStep::Unbounded),
        (Some(json!(0.5)), MaxDeltaStep::Bounded(0.5)),
    ] {
        let mut pairs = json!({"objective": "count:poisson"});
        if let Some(value) = value.clone() {
            pairs["max_delta_step"] = value;
        }
        let p = parse(pairs).unwrap();
        assert_eq!(p.max_delta_step, step, "{value:?}");
        let back = TrainingParams::from_xgboost(p.to_xgboost().unwrap()).unwrap();
        assert_eq!(back, p, "{value:?}");
    }
    let unlimited = TrainingParams {
        max_depth: None,
        grow_policy: GrowPolicy::LossGuide,
        max_leaves: NonZeroUsize::new(5),
        ..TrainingParams::default()
    };
    let flat = unlimited.to_xgboost().unwrap();
    assert_eq!(
        (flat["max_depth"].clone(), flat["nthread"].clone()),
        (json!(0), json!(0))
    );
    assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), unlimited);
    let negative = parse(json!({"max_delta_step": -1.0})).unwrap_err();
    assert!(
        negative
            .to_string()
            .starts_with("invalid parameter `max_delta_step`")
    );
}

/// Every built-in objective's flat form (its name and the keys of its
/// own parameters) reads back to the same objective.
#[test]
fn typed_objectives_round_trip_through_the_flat_form() {
    let dist = Distributional::new(DistFamily::Gamma).with_gradient(DistGradient::Natural);
    let objectives = [
        Objective::SquaredError(RegLoss::new(2.0).unwrap()),
        Objective::SquaredLogError,
        Objective::PseudoHuber(PseudoHuber::new(0.4).unwrap()),
        Objective::AbsoluteError,
        Objective::Quantile(Quantiles::new([0.1, 0.5, 0.9]).unwrap()),
        Objective::Expectile(Expectiles::new([0.2, 0.8]).unwrap()),
        Objective::RegLogistic(RegLoss::new(3.0).unwrap()),
        Objective::BinaryLogistic(RegLoss::new(2.5).unwrap()),
        Objective::BinaryLogitRaw(RegLoss::new(0.5).unwrap()),
        Objective::BinaryHinge,
        Objective::Softmax(Multiclass::new(4).unwrap()),
        Objective::Softprob(Multiclass::new(3).unwrap()),
        Objective::Poisson,
        Objective::Gamma(RegLoss::new(4.0).unwrap()),
        Objective::Tweedie(Tweedie::new(1.3).unwrap()),
        Objective::RankPairwise(LambdaRank::new(4).unwrap()),
        Objective::RankNdcg(LambdaRank::new(8).unwrap()),
        Objective::RankMap(LambdaRank::default()),
        Objective::RankXendcg,
        Objective::Cox,
        Objective::Aft(Aft::new(AftDistribution::Logistic, 1.7).unwrap()),
        Objective::Dist(dist),
    ];
    for objective in objectives {
        let p = TrainingParams {
            objective: objective.clone(),
            ..TrainingParams::default()
        };
        let flat = p.to_xgboost().unwrap();
        assert_eq!(flat["objective"], json!(objective.name()));
        let back = TrainingParams::from_xgboost(flat).unwrap();
        assert_eq!(back.objective, objective, "{}", objective.name());
        assert_eq!(back, p, "{}", objective.name());
    }
    // A split direction needs shared vector-leaf trees.
    let p = TrainingParams {
        objective: Objective::Dist(dist.with_split_direction(DistSplitDirection::Cyclic)),
        multi_strategy: MultiStrategy::MultiOutputTree,
        ..TrainingParams::default()
    };
    let flat = p.to_xgboost().unwrap();
    assert_eq!(flat["dist_split_direction"], json!("cyclic"));
    assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), p);
}

/// An objective-parameter key the objective does not read is refused by
/// name, unless a configured metric borrows it (as XGBoost's metrics
/// read the objective's parameters).
#[test]
fn objective_keys_nothing_reads_are_refused_by_name() {
    for (pairs, key) in [
        (
            json!({"objective": "binary:logistic", "num_class": 3}),
            "num_class",
        ),
        (
            json!({"objective": "reg:squaredlogerror", "scale_pos_weight": 2.0}),
            "scale_pos_weight",
        ),
        (
            json!({"objective": "count:poisson", "scale_pos_weight": 2.0}),
            "scale_pos_weight",
        ),
        (
            json!({"objective": "reg:gamma", "dist_gradient": "hessian"}),
            "dist_gradient",
        ),
        (
            json!({"objective": "reg:squarederror", "huber_slope": 0.5}),
            "huber_slope",
        ),
        (
            json!({"objective": "multi:softprob", "num_class": 3, "tweedie_variance_power": 1.2}),
            "tweedie_variance_power",
        ),
    ] {
        let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
        assert!(
            refusal.starts_with(&format!("invalid parameter `{key}`")),
            "{pairs}: {refusal}"
        );
    }
    let p = TrainingParams::from_xgboost([
        ("objective", json!("reg:squarederror")),
        ("eval_metric", json!("mphe")),
        ("huber_slope", json!(0.5)),
    ])
    .unwrap();
    assert_eq!(p.objective, Objective::SquaredError(RegLoss::default()));
    assert_eq!(
        p.eval_metric,
        [EvalMetric::Mphe(PseudoHuber::new(0.5).unwrap())]
    );
}

/// XGBoost's metrics read the flat parameters whatever the objective:
/// `mphe` the `huber_slope`, `quantile` / `expectile` the alpha lists,
/// `aft-nloglik` the AFT noise, and `nll` / `crps` the `dist:*` family.
/// A metric with other parameters has no flat form.
/// The fields are public, so a configuration can be invalid: it is
/// refused by name rather than written as a flat form that reads back
/// as a different one (a NaN bound or base score as `null`, i.e. unset).
#[test]
fn invalid_configurations_are_refused_not_serialized() {
    for (p, key) in [
        (
            TrainingParams {
                max_delta_step: MaxDeltaStep::Bounded(f64::NAN),
                ..TrainingParams::default()
            },
            "max_delta_step",
        ),
        (
            TrainingParams {
                base_score: Some(f64::NAN),
                ..TrainingParams::default()
            },
            "base_score",
        ),
    ] {
        match p.to_xgboost() {
            Err(HessboostError::InvalidParameter { name, .. }) => assert_eq!(name, key),
            other => panic!("{key}: expected a refusal, got {other:?}"),
        }
    }
}

/// A Tweedie metric's variance power survives the flat form in full:
/// its `evals_result` key rounds to six digits, its flat spelling not.
#[test]
fn tweedie_metric_powers_round_trip_exactly() {
    for power in ["1.999999", "1.23456789", "1.5"] {
        let p = TrainingParams::from_xgboost([(
            "eval_metric",
            json!(format!("tweedie-nloglik@{power}")),
        )])
        .unwrap();
        let back = TrainingParams::from_xgboost(p.to_xgboost().unwrap())
            .unwrap_or_else(|e| panic!("{power}: {e}"));
        assert_eq!(back.eval_metric, p.eval_metric, "{power}");
    }
}

#[test]
fn metrics_take_the_flat_parameters_xgboost_gives_them() {
    use crate::objective::distributional::DistFamily;
    let p = TrainingParams::from_xgboost([
        ("huber_slope", json!(0.7)),
        ("quantile_alpha", json!([0.2, 0.8])),
        ("aft_loss_distribution", json!("logistic")),
        (
            "eval_metric",
            json!(["mphe", "quantile", "aft-nloglik", "ndcg@3"]),
        ),
    ])
    .unwrap();
    assert_eq!(
        p.eval_metric,
        [
            EvalMetric::Mphe(PseudoHuber::new(0.7).unwrap()),
            EvalMetric::Quantile(Quantiles::new([0.2, 0.8]).unwrap()),
            EvalMetric::AftNLogLik(Aft::with_distribution(AftDistribution::Logistic)),
            EvalMetric::Ndcg(crate::metric::Cutoff::top(3).unwrap()),
        ]
    );
    let flat = p.to_xgboost().unwrap();
    assert_eq!(
        flat["eval_metric"],
        json!(["mphe", "quantile", "aft-nloglik", "ndcg@3"])
    );
    assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), p);

    // One flat `huber_slope` serves the objective and every metric, so
    // metrics or an objective with another slope have no flat form.
    let slope = |s| PseudoHuber::new(s).unwrap();
    let mut other_slope = p.clone();
    other_slope.eval_metric = vec![EvalMetric::Mphe(slope(2.0))];
    let flat = other_slope.to_xgboost().unwrap();
    assert_eq!(flat["huber_slope"], json!(2.0));
    assert_eq!(TrainingParams::from_xgboost(flat).unwrap(), other_slope);
    other_slope.eval_metric.push(EvalMetric::Mphe(slope(0.7)));
    assert!(other_slope.to_xgboost().is_err());
    let huber = TrainingParams {
        objective: Objective::PseudoHuber(slope(0.7)),
        eval_metric: vec![EvalMetric::Mphe(slope(2.0))],
        ..TrainingParams::default()
    };
    assert!(huber.to_xgboost().is_err());
    let no_dist = TrainingParams {
        eval_metric: vec![EvalMetric::Nll(DistFamily::Normal)],
        ..TrainingParams::default()
    };
    assert!(no_dist.to_xgboost().is_err());

    assert!(refused(json!({"eval_metric": "nll"})).is_some());
    assert!(refused(json!({"eval_metric": "quantile"})).is_some());
    let dist = TrainingParams::from_xgboost([
        ("objective", json!("dist:normal")),
        ("eval_metric", json!(["nll", "crps"])),
    ])
    .unwrap();
    assert_eq!(
        dist.eval_metric,
        [
            EvalMetric::Nll(DistFamily::Normal),
            EvalMetric::Crps(DistFamily::Normal)
        ]
    );
}

/// Nothing is ignored: unknown keys (with a suggestion for typos), a key
/// set twice through an alias, `null` for a plain value, `missing`,
/// one-setting options at another setting, and invalid configurations.
#[test]
fn unsupported_settings_are_refused_by_name() {
    for (pairs, message) in [
        (
            json!({"max_dept": 3}),
            "unknown parameter `max_dept` (did you mean `max_depth`?)",
        ),
        (json!({"zzz": 1}), "unknown parameter `zzz`"),
        (
            json!({"eta": 0.1, "learning_rate": 0.2}),
            "invalid parameter `eta`: is set twice",
        ),
        (
            json!({"eta": null}),
            "invalid parameter `eta`: invalid type: null",
        ),
        (
            json!({"eta": "0.1"}),
            "invalid parameter `eta`: invalid type: string",
        ),
        (json!({"missing": 0.0}), "invalid parameter `missing`"),
        (
            json!({"lambdarank_pair_method": "mean"}),
            r#"invalid parameter `lambdarank_pair_method`: is only implemented as "topk""#,
        ),
        (json!({"updater": "coord_descent"}), "gblinear's updater"),
        (json!({"monotone_constraints": "(1,x)"}), "bad entry `x`"),
        (json!({"monotone_constraints": [2]}), "-1, 0 or 1"),
        (json!({"subsample": 1.5}), "invalid parameter `subsample`"),
    ] {
        let refusal = refused(pairs.clone()).unwrap_or_else(|| panic!("{pairs} accepted"));
        assert!(refusal.contains(message), "{pairs}: {refusal}");
    }
    assert_eq!(
        refused(json!({"zzz": 1})).unwrap(),
        "unknown parameter `zzz`"
    );
    assert!(refused(json!({"booster": "gblinear", "updater": "coord_descent"})).is_none());
    assert!(refused(json!({"base_score": null, "max_delta_step": null})).is_none());
}

/// Every registered objective-parameter key is a flat key, so its refusal
/// check can see it set.
#[test]
fn objective_params_are_flat_keys() {
    for param in crate::objective::OBJECTIVE_PARAMS {
        assert!(super::schema::KEYS.contains(&param.key), "{}", param.key);
    }
}
