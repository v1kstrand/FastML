import os
import sys

import numpy as np
import pytest

sys.path.append(os.getcwd())
from module.grid_result import GridResult

# TEST GridResults CLASS


@pytest.fixture
def setup_grid_results():
    score_dict = {"mae": (np.inf, "", "<"), "r2": (-np.inf, "", ">")}
    return GridResult(score_dict)


def test_add_grid_result(setup_grid_results):
    setup_grid_results.add_grid_result("grid", "model_type")
    assert setup_grid_results.grid_results["model_type"] == "grid"


def test_update_score_dict(setup_grid_results):
    setup_grid_results.update_score_dict([10, 0.5], "SVR")

    assert setup_grid_results.score_dict["mae"][0] == 10
    assert setup_grid_results.score_dict["mae"][1] == "SVR"

    assert setup_grid_results.score_dict["r2"][0] == 0.5
    assert setup_grid_results.score_dict["r2"][1] == "SVR"

    setup_grid_results.update_score_dict([5, 0.6], "LinearRegression")
    assert setup_grid_results.score_dict["mae"][0] == 5
    assert setup_grid_results.score_dict["mae"][1] == "LinearRegression"

    assert setup_grid_results.score_dict["r2"][0] == 0.6
    assert setup_grid_results.score_dict["r2"][1] == "LinearRegression"

    setup_grid_results.update_score_dict([15, 0.4], "SVR")
    assert setup_grid_results.score_dict["mae"][0] == 5
    assert setup_grid_results.score_dict["mae"][1] == "LinearRegression"

    assert setup_grid_results.score_dict["r2"][0] == 0.6
    assert setup_grid_results.score_dict["r2"][1] == "LinearRegression"


@pytest.fixture
def setup_grid_results_with_score(setup_grid_results):
    setup_grid_results.update_score_dict([10, 0.5], "SVR")
    return setup_grid_results


def test_evaluate_models_and_get_best(setup_grid_results):
    setup_grid_results.add_grid_result("SVR", "SVR")
    setup_grid_results.add_grid_result("LinearRegression", "LinearRegression")
    setup_grid_results.update_score_dict([5, 0.8], "SVR")
    setup_grid_results.update_score_dict([10, 0.5], "LinearRegression")
    assert setup_grid_results.evaluate_models_and_get_best() == "SVR"

    setup_grid_results.add_grid_result("ANN", "ANN")
    setup_grid_results.update_score_dict([4, 0.9], "ANN")
    assert setup_grid_results.evaluate_models_and_get_best() == "ANN"


def test_display_results(setup_grid_results_with_score, capsys):
    setup_grid_results_with_score.display_results("ANN")
    captured = capsys.readouterr()
    assert captured.out.startswith("Best mae - score: 10.00 from SVR")


if __name__ == "__main__":
    pytest.main([__file__])
