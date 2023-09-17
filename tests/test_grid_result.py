import os
import sys
from unittest.mock import Mock

import pytest

sys.path.append(os.getcwd())
from module.grid_result import GridResult

# TEST GridResults CLASS


@pytest.fixture
def setup_grid_results():
    metrics = ["mae", "r2"]
    return GridResult(metrics)


# Test add_grid_result method


def test_add_grid_result(setup_grid_results):
    setup_grid_results.add_grid_result("grid", "model_type")
    assert setup_grid_results.grid_results["model_type"] == "grid"


# Test update_score_dict method
def test_update_score_dict(setup_grid_results):
    setup_grid_results.update_score_dict({"mae": 10, "r2": 0.5}, "SVR")

    assert setup_grid_results.score_dict["SVR"]["mae"] == 10
    assert setup_grid_results.score_dict["SVR"]["r2"] == 0.5


@pytest.fixture
def setup_grid_with_score(setup_grid_results):
    setup_grid_results.update_score_dict({"mae": 10, "r2": 0.5}, "SVR")
    setup_grid_results.update_score_dict({"mae": 9, "r2": 0.6}, "ANN")
    setup_grid_results.add_grid_result("grid", "ANN")
    return setup_grid_results


# Test let_user_pick_preferred_model method
def test_let_user_pick_preferred_model(setup_grid_with_score):
    assert setup_grid_with_score.let_user_pick_preferred_model(
        lambda x, y: 1
    ) == ("grid", "ANN")


def test_let_user_pick_preferred_model_func_call(setup_grid_with_score):
    mock_func = Mock()
    setup_grid_with_score.display_results = mock_func
    setup_grid_with_score.display_suggested_model = mock_func
    # User will pick model 1 each time (lambda x, y: 1)
    setup_grid_with_score.let_user_pick_preferred_model(lambda x, y: 1)
    assert mock_func.call_count == 2


# Test display_results method
def test_display_results(setup_grid_with_score, capsys):
    setup_grid_with_score.display_results()
    captured = capsys.readouterr()
    assert captured.out.endswith("0.60\n")


# Test display_suggested_model method
def test_display_suggested_model(setup_grid_with_score, capsys):
    setup_grid_with_score.display_suggested_model()
    captured = capsys.readouterr()
    assert captured.out.endswith("ANN\n")


if __name__ == "__main__":
    pytest.main([__file__])
