import os
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import sklearn.base
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

sys.path.append(os.getcwd())
from module.grid_search import GridSearch


# TEST GridSearch class
@pytest.fixture
def setup_data():
    X, y = make_classification(n_samples=100, n_features=20)
    return (X, X, y, y)


@pytest.fixture
def setup_grid(setup_data):
    X, y = make_classification(n_samples=100, n_features=20)
    models = {LogisticRegression(): {"C": [1, 10]}}
    return GridSearch(setup_data, Mock(), Mock(), models)


# Test execute method
def test_execute(setup_grid):
    setup_grid.process_model = Mock()
    assert setup_grid.execute() is None
    assert setup_grid.process_model.call_count == 1


# Test process_model method
def test_process_model(setup_grid):
    model = list(setup_grid.models.keys())[0]
    assert setup_grid.process_model(model, "LogisticRegression") is None

    preds, model_type = setup_grid.evaluator_callback.call_args[0]
    assert isinstance(preds, np.ndarray)
    assert model_type == "LogisticRegression"

    grid, model_type = setup_grid.grid_results_callback.call_args[0]
    assert isinstance(grid, GridSearchCV)
    assert model_type == "LogisticRegression"


# Test perform_grid_search method
def test_perform_grid_search(setup_grid):
    model = list(setup_grid.models.keys())[0]
    grid, preds = setup_grid.perform_grid_search(model)
    assert isinstance(grid, sklearn.base.BaseEstimator)
    assert isinstance(preds, np.ndarray)


# TEST GridSearch class with two models
@pytest.fixture
def setup_two_models(setup_data):
    models = {
        LogisticRegression(): {"C": [1, 10]},
        SVC(): {"C": [0.01, 1, 100], "gamma": ["scale", "auto"]},
    }
    return GridSearch(setup_data, Mock(), Mock(), models)


def test_execute_two_models(setup_two_models):
    setup_two_models.process_model = Mock()
    assert setup_two_models.execute() is None
    assert setup_two_models.process_model.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])
