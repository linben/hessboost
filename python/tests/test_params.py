"""The parameter mapping: XGBoost names and aliases, refusals, and the
feature-name forms of the constraint parameters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import hessboost
from conftest import regression
from hessboost import DMatrix, HessboostError


@pytest.fixture(scope="module")
def dtrain() -> DMatrix:
    x, y = regression(rows=200)
    return DMatrix(x, y, feature_names=["a", "b", "c", "d", "e"])


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"max_dept": 3}, r"unknown parameter `max_dept` \(did you mean `max_depth`\?\)"),
        ({"colsample_by_tree": 0.5}, r"did you mean `colsample_bytree`"),
        ({"zzz": 1}, r"^unknown parameter `zzz`$"),
        ({"eta": "0.1"}, r"parameter `eta`: invalid type: string"),
        ({"max_depth": -1}, r"parameter `max_depth`"),
        ({"max_depth": 2.5}, r"parameter `max_depth`"),
        ({"tree_method": "gpu_hist"}, r"unknown variant `gpu_hist`"),
        ({"device": "cuda"}, r"`cuda` requires building with the `cuda` feature"),
        ({"device": "cuda:x"}, r"unknown variant `cuda:x`"),
        ({"eta": 0.1, "learning_rate": 0.2}, r"`eta`: is set twice"),
        ({"subsample": 1.5}, r"subsample"),
        ({"eta": float("nan")}, r"`eta` must be finite"),
        ({"missing": 0.0}, r"DMatrix\(data, missing=\.\.\.\)"),
        ({"lambdarank_pair_method": "mean"}, r"only implemented as \"topk\""),
        ({"max_cat_to_onehot": 8}, r"only implemented as 4"),
        ({"updater": "coord_descent"}, r"gblinear's updater"),
        ({"verbosity": 0}, r"unknown parameter `verbosity`"),
        ({"objective": "reg:nonsense"}, r"reg:nonsense"),
        ({"monotone_constraints": "(1,x)"}, r"bad entry `x`"),
        ({"monotone_constraints": [2]}, r"-1, 0 or 1"),
    ],
)
def test_bad_parameters_are_refused_by_name(
    dtrain: DMatrix, params: dict[str, Any], message: str
) -> None:
    with pytest.raises(HessboostError, match=message):
        hessboost.train(params, dtrain, 1)


def test_parameters_must_be_a_mapping_of_plain_values(dtrain: DMatrix) -> None:
    with pytest.raises(TypeError, match="mapping"):
        hessboost.train([("eta", 0.1)], dtrain, 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="parameter `eta` has unsupported type object"):
        hessboost.train({"eta": object()}, dtrain, 1)


def test_aliases_and_numpy_values_set_the_same_model(dtrain: DMatrix) -> None:
    canonical = hessboost.train(
        {"eta": 0.2, "lambda": 2.0, "alpha": 0.5, "gamma": 0.1, "seed": 3, "nthread": 2},
        dtrain,
        5,
    )
    aliased = hessboost.train(
        {
            "learning_rate": np.float32(0.2),
            "reg_lambda": np.float64(2.0),
            "reg_alpha": 0.5,
            "min_split_loss": 0.1,
            "random_state": np.int64(3),
            "n_jobs": 2,
        },
        dtrain,
        5,
    )
    assert canonical.save_raw() == aliased.save_raw()
    other = hessboost.train({"eta": 0.2}, dtrain, 5)
    assert other.save_raw() != canonical.save_raw()


def test_balanced_bagging_params_are_accepted_and_validated() -> None:
    x = np.linspace(-2, 2, 400).reshape(-1, 1)
    y = (x[:, 0] > 0).astype(np.float32)
    dtrain = DMatrix(x, y)
    params = {
        "objective": "binary:logistic",
        "pos_bagging_fraction": 0.7,
        "neg_bagging_fraction": 0.3,
        "subsample": 1.0,
        "seed": 9,
    }
    first = hessboost.train(params, dtrain, 4)
    second = hessboost.train(params, dtrain, 4)
    assert first.save_raw() == second.save_raw()
    with pytest.raises(HessboostError, match="balanced bagging needs a `binary:\\*` objective"):
        hessboost.train({"objective": "reg:squarederror", "pos_bagging_fraction": 0.5}, dtrain, 1)
    with pytest.raises(HessboostError, match="balanced bagging replaces `subsample`"):
        hessboost.train(
            {
                "objective": "binary:logistic",
                "pos_bagging_fraction": 0.7,
                "neg_bagging_fraction": 0.3,
                "subsample": 0.9,
            },
            dtrain,
            1,
        )
    with pytest.raises(
        HessboostError,
        match="query bagging is not supported together with class-balanced bagging",
    ):
        hessboost.train(
            {
                "objective": "rank:ndcg",
                "bagging_by_query": True,
                "pos_bagging_fraction": 0.5,
                "subsample": 0.5,
            },
            DMatrix(np.arange(8.0).reshape(4, 2), [0, 1, 0, 1], group=[2, 2]),
            1,
        )


def test_fixed_options_are_accepted_at_their_only_setting(dtrain: DMatrix) -> None:
    hessboost.train({"max_cat_to_onehot": 4, "max_cat_threshold": 64}, dtrain, 1)
    hessboost.train(
        {"objective": "rank:ndcg", "lambdarank_pair_method": "topk"},
        DMatrix(np.arange(8.0).reshape(4, 2), [0, 1, 0, 1], group=[2, 2]),
        1,
    )


@pytest.mark.parametrize(
    "constraints",
    ["(1,0,0,0,0)", [1, 0, 0, 0, 0], (1, 0, 0, 0, 0), {"a": 1}, {0: 1}],
)
def test_monotone_constraints_in_every_xgboost_form(dtrain: DMatrix, constraints: object) -> None:
    booster = hessboost.train({"monotone_constraints": constraints}, dtrain, 20)
    grid = np.zeros((50, 5))
    grid[:, 0] = np.linspace(-3, 3, 50)
    assert np.all(np.diff(booster.predict(grid)) >= 0)


def test_monotone_constraints_by_name_need_known_features(dtrain: DMatrix) -> None:
    with pytest.raises(HessboostError, match="unknown feature 'z'"):
        hessboost.train({"monotone_constraints": {"z": 1}}, dtrain, 1)


def test_interaction_constraints_by_index_name_and_string(dtrain: DMatrix) -> None:
    by_index = hessboost.train({"interaction_constraints": [[0, 1], [2, 3, 4]]}, dtrain, 5)
    by_name = hessboost.train({"interaction_constraints": [["a", "b"], ["c", "d", "e"]]}, dtrain, 5)
    by_string = hessboost.train({"interaction_constraints": "[[0, 1], [2, 3, 4]]"}, dtrain, 5)
    assert by_index.save_raw() == by_name.save_raw() == by_string.save_raw()
    assert by_index.save_raw() != hessboost.train({}, dtrain, 5).save_raw()
