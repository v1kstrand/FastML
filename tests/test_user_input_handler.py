import os
import sys
from unittest.mock import Mock

import pandas as pd
import pytest

sys.path.append(os.getcwd())
from module.user_input_handler import InputValidator, UserInputHandler

# TEST UserInputHandler CLASS


def test_build_up():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv("tmp_mock_test.csv", index=False)


class MockFileHandler:
    @staticmethod
    def read():
        return (
            "# task_type - -> regression\n"
            "# csv_path - -> tmp_mock_test.csv\n"
            "# target - -> target    \n"
            "# split_size - -> 0.5\n"
            "# model_name - -> model_name\n"
            "# scale - -> zscore\n"
            "# drop_columns - ->  \n"
        )


@pytest.fixture
def valid_input():
    return {
        "task_type": "regression",
        "csv_path": "tmp_mock_test.csv",
        "target": "target",
        "split_size": "0.5",
        "model_name": "model_name",
        "scale": "zscore",
        "drop_columns": "",
    }


@pytest.fixture
def setup_handler():
    return UserInputHandler(MockFileHandler.read, Mock())


# Test process method
def test_process(setup_handler):
    assert setup_handler.process() == setup_handler.user_input
    assert setup_handler.user_input["task_type"] == False
    assert setup_handler.user_input["csv_path"] == "tmp_mock_test.csv"
    assert setup_handler.user_input["target"] == "target"
    assert setup_handler.user_input["split_size"] == 0.5
    assert setup_handler.user_input["model_name"] == "model_name"
    assert setup_handler.user_input["drop_columns"] == []


# Test extract_user_input_from_file method
def test_extract_user_input_from_file(setup_handler):
    user_input = "# target - -> price\n# path - csv_path -> tmp_mock_test.csv"
    setup_handler.extract_user_input(user_input)
    assert setup_handler.user_input == {
        "target": "price",
        "path": "tmp_mock_test.csv",
    }


def test_extract_user_input_strip(setup_handler):
    strip_input = "# target - -> house price    \n"
    setup_handler.extract_user_input(strip_input)
    assert setup_handler.user_input == {"target": "house price"}


# transform_user_input
def test_transform_user_input(valid_input, setup_handler):
    setup_handler.user_input = valid_input
    setup_handler.transform_user_input()
    inp = setup_handler.user_input
    assert inp["task_type"] == False
    assert inp["csv_path"] == "tmp_mock_test.csv"
    assert inp["target"] == "target"
    assert inp["split_size"] == 0.5 and isinstance(inp["split_size"], float)
    assert inp["model_name"] == "model_name"
    assert inp["drop_columns"] == []


def test_transform_user_input_drop_column(valid_input, setup_handler):
    setup_handler.user_input = valid_input
    setup_handler.user_input["drop_columns"] = "a"
    setup_handler.transform_user_input()
    assert setup_handler.user_input["drop_columns"] == ["a"]


def test_transform_user_input_drop_columns(valid_input, setup_handler):
    setup_handler.user_input = valid_input
    setup_handler.user_input["drop_columns"] = "a|b|c"
    setup_handler.transform_user_input()
    assert setup_handler.user_input["drop_columns"] == ["a", "b", "c"]


# # TEST InputValidator CLASS


# Test validate method
def test_validate(valid_input):
    assert InputValidator.validate(valid_input) is None


# Test validate_keys method
def test_validate_keys(valid_input):
    assert InputValidator.validate_keys(valid_input) is None


def test_validate_keys_invalid_task_type(valid_input):
    valid_input.pop("task_type")
    with pytest.raises(ValueError):
        InputValidator.validate_keys(valid_input)


# Test validate_values method
def test_validate_values(valid_input):
    assert InputValidator.validate_values(valid_input) is None


def test_validate_values_invalid(valid_input):
    valid_input["task_type"] = ""
    with pytest.raises(ValueError):
        InputValidator.validate_values(valid_input)


# Test validate_rules method
def test_validate_rules(valid_input):
    assert InputValidator.validate_rules(valid_input) is None


def test_validate_rules_invalid_csv_path(valid_input):
    valid_input["csv_path"] = "invalid_path"
    with pytest.raises(FileNotFoundError):
        InputValidator.validate_rules(valid_input)


def test_validate_rules_invalid_task_type(valid_input):
    valid_input["task_type"] = "invalid"
    with pytest.raises(AssertionError):
        InputValidator.validate_rules(valid_input)


def test_validate_rules_invalid_scale(valid_input):
    valid_input["scale"] = "invalid"
    with pytest.raises(AssertionError):
        InputValidator.validate_rules(valid_input)


def test_validate_rules_invalid_split_size(valid_input):
    valid_input["split_size"] = "2"
    with pytest.raises(AssertionError):
        InputValidator.validate_rules(valid_input)


def test_teardown_module():
    if os.path.exists("tmp_mock_test.csv"):
        os.remove("tmp_mock_test.csv")


if __name__ == "__main__":
    pytest.main([__file__])
