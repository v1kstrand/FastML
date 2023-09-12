import os
import sys
from unittest.mock import Mock, patch

sys.path.append(os.getcwd())

import pandas as pd
import pytest

from module.main_util import (
    handle_user_input,
    perform_grid_search,
    prepare_models_and_grid,
    save_best_model,
    transform_data,
)


@pytest.fixture
def user_input():
    return {
        "task_type": True,
        "csv_path": "tmp_mock_test.csv",
        "target": "target",
        "split_size": 0.5,
        "model_name": "model_name",
        "scale": "zscore",
        "drop_columns": "",
    }


@pytest.fixture
def setup_df():
    return pd.DataFrame(
        {
            "col1": list(range(1000)),
            "col2": list(range(1000)),
            "target": [1] * 1000,
        }
    )


def test_handle_user_input():
    with patch("module.main_util.UserInputHandler") as UserInputHandler:
        UserInputHandler.return_value.process.return_value = "mock_output"
        assert handle_user_input() == "mock_output"


def test_transform_data(user_input, setup_df):
    setup_df.to_csv("tmp_mock_test.csv", index=False)
    data, num_features, num_classes = transform_data(user_input)
    assert len(data) == 4
    assert num_features == 2
    assert num_classes == 1
    os.remove("tmp_mock_test.csv")


def test_prepare_models_and_grid():
    models, grid_result, evaluator = prepare_models_and_grid(
        True, [1, 2, 3], 2, 1
    )
    assert len(models) == 4
    assert grid_result.grid_results == {}
    assert evaluator.y_test == [1, 2, 3]


def test_perform_grid_search():
    """
    The function create some (mock) input data to call
    the perform_grid_search function with.

    Then we mock the GridSearch and creates a GridSearch object by
    calling perform_grid_search with the generated input data.

    We then call the execute method of the mock GridSearch object.

    Finally, we check if the mock GridSearch was created
    with the correct arguments and that the execute method
    was called once.
    """
    mock_evaluator = Mock()
    mock_grid_result = Mock()
    mock_models = {}
    mock_data = [1] * 4

    with patch("module.main_util.GridSearch") as MockGridSearch:
        perform_grid_search(
            mock_data, mock_evaluator, mock_grid_result, mock_models
        )
        MockGridSearch.assert_called_once_with(
            mock_data,
            mock_evaluator.evaluate,
            mock_grid_result.add_grid_result,
            mock_models,
        )
        MockGridSearch.return_value.execute.assert_called_once()


def test_save_best_model():
    """
    Turn the GridResult class into a mock and call save_best_model
    with the mock GridResult class and a mock model name.

    Then the mock_grid_result.evaluate_models_and_get_best method
    is modified to return a string "Best Model".

    The save_best_model function will then call the mock_grid_result
    and save the returned string to a file called "mock_name.joblib".

    Finally, we check if the file exists and remove it.

    """
    mock_grid_result = Mock()
    mock_grid_result.evaluate_models_and_get_best.return_value = "Best Model"

    save_best_model(mock_grid_result, "mock_name")
    assert os.path.exists(r"models\mock_name.joblib")
    os.remove(r"models\mock_name.joblib")


if __name__ == "__main__":
    pytest.main([__file__])
