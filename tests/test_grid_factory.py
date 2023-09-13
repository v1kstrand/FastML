import os
import sys

import numpy as np
import pytest

sys.path.append(os.getcwd())
from module.evaluators import ClassifierEvaluator, RegressorEvaluator
from module.grid_factory import GridFactory
from module.grid_result import GridResult


@pytest.fixture
def setup_y_test():
    return np.array([0, 1, 1, 0, 1])


def test_grid_factory_build_classifier(setup_y_test):
    # Classifier
    grid_result, evaluator = GridFactory.build(True, setup_y_test)

    assert len(grid_result.metrics) == 4
    assert isinstance(grid_result, GridResult)
    assert isinstance(evaluator, ClassifierEvaluator)

    # Regressor
    grid_result, evaluator = GridFactory.build(False, setup_y_test)

    assert len(grid_result.metrics) == 3
    assert isinstance(grid_result, GridResult)
    assert isinstance(evaluator, RegressorEvaluator)


def test_classifier_build(setup_y_test):
    grid_result, evaluator = GridFactory.classifier_build(setup_y_test)

    assert isinstance(grid_result, GridResult)
    assert isinstance(evaluator, ClassifierEvaluator)
    assert grid_result.metrics == ["precision", "recall", "f1", "accuracy"]
    assert (
        evaluator.grid_results_callback.__func__
        is grid_result.update_score_dict.__func__
    )


def test_regressor_build(setup_y_test):
    factory = GridFactory()
    grid_result, evaluator = factory.regressor_build(setup_y_test)

    assert isinstance(grid_result, GridResult)
    assert isinstance(evaluator, RegressorEvaluator)

    assert grid_result.metrics == ["mae", "rmse", "r2"]
    assert (
        evaluator.grid_results_callback.__func__
        is grid_result.update_score_dict.__func__
    )


if __name__ == "__main__":
    pytest.main([__file__])
