import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.getcwd())

import pandas as pd
import pytest

from module.utils import (
    CLASSIFICATION_THRESHOLD,
    FileHandler,
    get_user_choice,
    read_data,
    save_model,
    validate_data_type,
)


# Test write_file method
def test_write_file():
    FileHandler.write_file("mock_input.txt")
    assert Path("mock_input.txt").is_file()
    os.remove("mock_input.txt")


# Test read_file method
def test_read_file():
    FileHandler.write_file(r"user_input\mock_input.txt")
    assert FileHandler.read(lambda x, y: 0).startswith("Complete this form")
    os.remove(r"user_input\mock_input.txt")


# Test get_user_input_files method
def test_get_user_input_files():
    with patch("module.utils.FileHandler.validate_user_path") as mock_validate:
        FileHandler.get_user_input_files("mock_folder")
        mock_validate.assert_called_once_with("mock_folder")


# Test validate_user_path method


def test_validate_user_path():
    os.mkdir("tmp_mock_folder")
    FileHandler.write_file(r"tmp_mock_folder\mock_input.txt")
    user_filer_list = FileHandler.get_user_input_files("tmp_mock_folder")
    assert user_filer_list == ["mock_input.txt"]
    os.remove(r"tmp_mock_folder\mock_input.txt")
    os.rmdir("tmp_mock_folder")


def test_validate_user_path_empty_folder():
    with patch("module.utils.FileHandler.write_file"):
        with pytest.raises(FileNotFoundError):
            FileHandler.get_user_input_files("tmp_mock_folder")
    os.rmdir("tmp_mock_folder")


# Test get_user_choice function
def test_get_user_choice():
    with patch("module.utils.input", side_effect=["1"]):
        assert get_user_choice(10) == 0


def test_get_user_choice_invalid_choice():
    with patch("module.utils.input", side_effect=["11", "12"]):
        with pytest.raises(ValueError):
            get_user_choice(10, "", 1)


def test_get_user_choice_valid_choice_after_invalid():
    with patch("module.utils.input", side_effect=["11", "5"]):
        assert get_user_choice(10, "", 1) == 4


# Test validate_data_type function
def test_validate_data_type_regression():
    assert validate_data_type(True, CLASSIFICATION_THRESHOLD)
    assert validate_data_type(False, CLASSIFICATION_THRESHOLD + 1)


def test_validate_data_type_strict_and_invalid_type():
    with pytest.raises(ValueError):
        validate_data_type(True, CLASSIFICATION_THRESHOLD + 1)
    with pytest.raises(ValueError):
        validate_data_type(False, CLASSIFICATION_THRESHOLD)


def test_validate_data_type_not_strict_and_invalid_type():
    assert not validate_data_type(
        False, CLASSIFICATION_THRESHOLD, strict=False
    )


# Test save_model function
class Model:
    pass


def test_save_model():
    save_model(Model(), "mock_model", "LogisticRegression")
    assert os.path.exists(r"models\mock_model_LogisticRegression.joblib")
    os.remove(r"models\mock_model_LogisticRegression.joblib")


# TEST read_data FUNCTION
def test_validate_data_df():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = read_data(df)
    assert isinstance(df, pd.DataFrame)


def test_validate_data_invalid_file_path():
    with pytest.raises(FileNotFoundError):
        read_data("invalid_file_path")


def test_validate_data_invalid_numpy():
    with pytest.raises(TypeError):
        read_data({"a": [1, 2, 3], "b": [4, 5, 6]})


if __name__ == "__main__":
    pytest.main([__file__])
