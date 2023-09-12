import os
import sys

import numpy as np
import pytest

sys.path.append(os.getcwd())
from module.evaluators import ClassifierEvaluator, RegressorEvaluator
from module.grid_factory import GridFactory
from module.grid_result import GridResult


def test_grid_factory_build_classifier():
    y_test = np.array([0, 1, 1, 0, 1])
    factory = GridFactory()

    # Classifier
    grid_result, evaluator = factory.build(True, y_test)

    assert len(grid_result.score_dict) == 4
    assert isinstance(grid_result, GridResult)
    assert isinstance(evaluator, ClassifierEvaluator)

    # Regressor
    grid_result, evaluator = factory.build(False, y_test)

    assert len(grid_result.score_dict) == 3
    assert isinstance(grid_result, GridResult)
    assert isinstance(evaluator, RegressorEvaluator)


def test_classifier_build():
    y_test = np.array([0, 1, 1, 0, 1])
    factory = GridFactory()

    grid_result, evaluator = factory.classifier_build(y_test)

    assert isinstance(grid_result, GridResult)
    assert isinstance(evaluator, ClassifierEvaluator)

    expected_score_dict = {
        "precision": (-np.inf, "", ">"),
        "recall": (-np.inf, "", ">"),
        "f1": (-np.inf, "", ">"),
        "accuracy": (-np.inf, "", ">"),
    }
    assert grid_result.score_dict == expected_score_dict
    assert evaluator.grid_results_callback == grid_result.update_score_dict


def test_regressor_build():
    y_test = np.array([0, 1, 1, 0, 1])
    factory = GridFactory()
    grid_result, evaluator = factory.regressor_build(y_test)

    assert isinstance(grid_result, GridResult)
    assert isinstance(evaluator, RegressorEvaluator)

    expected_score_dict = {
        "mae": (np.inf, "", "<"),
        "rmse": (np.inf, "", "<"),
        "r2": (-np.inf, "", ">"),
    }
    assert grid_result.score_dict == expected_score_dict
    assert evaluator.grid_results_callback == grid_result.update_score_dict


if __name__ == "__main__":
    pytest.main([__file__])
