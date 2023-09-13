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


# Test evaluate method
def test_evaluate_classifier(setup_classifier, setup_cls_preds):
    mock_func = Mock()
    setup_classifier.calculate_metrics = mock_func
    setup_classifier.display_score = mock_func
    setup_classifier.grid_results_callback = mock_func
    setup_classifier.reset_score = mock_func
    assert setup_classifier.evaluate(setup_cls_preds, "mondel_name") is None
    assert mock_func.call_count == 4


# Test calculate_metrics method
def test_calculate_metrics_classifier(setup_classifier, setup_cls_preds):
    setup_classifier.calculate_metrics(setup_cls_preds)
    assert len(setup_classifier.score) == 4
    assert isinstance(setup_classifier.score["precision"], float)
    assert isinstance(setup_classifier.score["recall"], float)
    assert isinstance(setup_classifier.score["f1"], float)
    assert isinstance(setup_classifier.score["accuracy"], float)


def test_calc_metrics_perfect_score_classifier(setup_classifier):
    preds = setup_classifier.y_test
    setup_classifier.calculate_metrics(preds)
    assert list(setup_classifier.score.values()) == [1.0, 1.0, 1.0, 1.0]


# Test reset_score method
def test_reset_score_classifier(setup_classifier, setup_cls_preds):
    setup_classifier.calculate_metrics(setup_cls_preds)
    setup_classifier.reset_score()
    assert setup_classifier.score == {}


# Test display_score method
def test_display_metrics_classifier(setup_classifier, setup_cls_preds, capsys):
    setup_classifier.display_score(setup_cls_preds)
    captured = capsys.readouterr()
    assert captured.out.startswith(
        "              precision    recall  f1-score   support"
    )
    assert captured.out.endswith("]]\n")


# # TEST RegressorEvaluator CLASS


@pytest.fixture
def setup_regressor():
    y = np.random.rand(100)
    return RegressorEvaluator(y, Mock())


@pytest.fixture
def setup_reg_preds():
    return np.random.rand(100)


# Test evaluate method
def test_evaluate_regressor(setup_regressor, setup_reg_preds):
    mock_func = Mock()
    setup_regressor.calculate_metrics = mock_func
    setup_regressor.grid_results_callback = mock_func
    setup_regressor.reset_score = mock_func
    assert setup_regressor.evaluate(setup_reg_preds, "mondel_name") is None
    assert mock_func.call_count == 3


# Test calculate_metrics method
def test_calculate_metrics_regressor(setup_regressor, setup_reg_preds):
    setup_regressor.calculate_metrics(setup_reg_preds)
    assert len(setup_regressor.score) == 3
    assert isinstance(setup_regressor.score["mae"], float)
    assert isinstance(setup_regressor.score["rmse"], float)
    assert isinstance(setup_regressor.score["r2"], float)


def test_calc_metrics_perfect_score_regressor(setup_regressor):
    preds = setup_regressor.y_test
    setup_regressor.calculate_metrics(preds)
    assert list(setup_regressor.score.values()) == [0.0, 0.0, 1.0]


# Test reset_score method
def test_reset_score_regressor(setup_regressor, setup_reg_preds):
    setup_regressor.calculate_metrics(setup_reg_preds)
    setup_regressor.reset_score()
    assert setup_regressor.score == {}


if __name__ == "__main__":
    pytest.main([__file__])
