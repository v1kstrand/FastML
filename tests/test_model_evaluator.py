import os
import sys
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.append(os.getcwd())
from module.evaluators import ClassifierEvaluator, RegressorEvaluator

# TEST ClassifierEvaluator CLASS


@pytest.fixture
def setup_classifier():
    y = np.random.randint(2, size=100)
    return ClassifierEvaluator(y, Mock())


@pytest.fixture
def setup_cls_preds():
    return np.random.randint(2, size=100)


def test_evaluate_classifier(setup_classifier, setup_cls_preds):
    mock_func = Mock()
    setup_classifier.calculate_metrics = mock_func
    setup_classifier.display_metrics = mock_func
    setup_classifier.grid_results_callback = mock_func
    assert setup_classifier.evaluate(setup_cls_preds, "mondel_name") is None
    assert mock_func.call_count == 3


def test_calculate_metrics_classifier(setup_classifier, setup_cls_preds):
    metrics = setup_classifier.calculate_metrics(setup_cls_preds)
    assert len(metrics) == 4
    assert isinstance(metrics[0], float)
    assert isinstance(metrics[1], float)
    assert isinstance(metrics[2], float)
    assert isinstance(metrics[3], float)


def test_calc_metrics_perfect_score_classifier(setup_classifier):
    preds = setup_classifier.y_test
    metrics = setup_classifier.calculate_metrics(preds)
    assert metrics == [1.0, 1.0, 1.0, 1.0]


def test_display_metrics_classifier(setup_classifier, setup_cls_preds, capsys):
    setup_classifier.display_metrics(setup_cls_preds)
    captured = capsys.readouterr()
    assert captured.out.startswith(
        "              precision    recall  f1-score   support"
    )


# # TEST RegressorEvaluator CLASS


@pytest.fixture
def setup_regressor():
    y = np.random.rand(100)
    return RegressorEvaluator(y, Mock())


@pytest.fixture
def setup_reg_preds():
    return np.random.rand(100)


def test_evaluate_regressor(setup_regressor, setup_reg_preds):
    mock_func = Mock()
    setup_regressor.calculate_metrics = mock_func
    setup_regressor.display_metrics = mock_func
    setup_regressor.grid_results_callback = mock_func
    assert setup_regressor.evaluate(setup_reg_preds, "mondel_name") is None
    assert mock_func.call_count == 3


def test_calculate_metrics_regressor(setup_regressor, setup_reg_preds):
    metrics = setup_regressor.calculate_metrics(setup_reg_preds)
    assert len(metrics) == 3
    assert isinstance(metrics[0], float)
    assert isinstance(metrics[1], float)
    assert isinstance(metrics[2], float)


def test_calc_metrics_perfect_score_regressor(setup_regressor):
    preds = setup_regressor.y_test
    metrics = setup_regressor.calculate_metrics(preds)
    assert metrics == [0.0, 0.0, 1.0]


def test_display_metrics_regressor(setup_regressor, capsys):
    setup_regressor.display_metrics([1, 1, 1])
    captured = capsys.readouterr()
    assert captured.out.startswith("MAE: ")


if __name__ == "__main__":
    pytest.main([__file__])
