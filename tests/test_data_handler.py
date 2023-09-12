import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.getcwd())
from module.data_handler import DataCleaner, DataTransformer, DataValidator


@pytest.fixture
def setup_data():
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.choice([0, 1], 100),
            "feature3": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }
    )

    return DataTransformer(df)


# TEST DATA TRANSFORMER CLASS


#  Test process_data method
def test_process_data(setup_data):
    X_train, X_test, y_train, y_test = setup_data.process_data(
        "target", "classification", 0.2, "zscore", ["feature1"]
    )
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


# Test set_x_y_data method
def test_set_x_y(setup_data):
    setup_data.set_x_y_data("target", ["feature1"])

    assert len(setup_data.X) == 100
    assert setup_data.y is not None
    assert "feature1" not in setup_data.X.columns


# Test set_correct_data_type method_and_shape method
def test_set_correct_data_type_regressor(setup_data):
    setup_data.set_x_y_data("target", ["feature1"])
    setup_data.set_correct_data_type_and_shape(False)
    assert setup_data.X.dtypes[0] == "float32"
    assert setup_data.X.shape == (100, 2)
    assert setup_data.y.dtype == "float32"
    assert setup_data.y.shape == (100, 1)


def test_set_correct_data_type_classifier(setup_data):
    setup_data.set_x_y_data("target")
    setup_data.set_correct_data_type_and_shape(True)
    assert setup_data.X.dtypes[0] == "float32"
    assert setup_data.y.dtype == "int64"
    assert setup_data.y.shape == (100,)


# Test split_scale_data method


@pytest.fixture
def setup_values(setup_data):
    setup_data.set_x_y_data("target")
    setup_data.X = setup_data.X.values
    return setup_data


def test_split_data_into_train_test(setup_values):
    split_size = 0.2
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = setup_values.split_data_into_train_test(split_size, True)
    assert len(X_train) == len(setup_values.X) * (1 - split_size)
    assert len(X_test) == len(setup_values.X) * split_size
    assert len(y_train) == len(setup_values.X) * (1 - split_size)
    assert len(y_test) == len(setup_values.X) * split_size


# Test scale_data_columns method


def test_scale_data_columns(setup_values):
    data = setup_values.split_data_into_train_test(0.2, True)
    X_train, *_ = setup_values.scale_data_columns(data, "zscore")

    assert np.isclose(X_train[:, 0].mean(), 0, atol=0.01)
    assert np.isclose(X_train[:, 0].std(), 1, atol=0.01)


def test_scale_data_columns_minmax(setup_values):
    data = setup_values.split_data_into_train_test(0.2, True)
    X_train, *_ = setup_values.scale_data_columns(data, "minmax")

    assert np.isclose(X_train[:, 0].min(), 0, atol=0.01)
    assert np.isclose(X_train[:, 0].max(), 1, atol=0.01)


def test_scale_data_columns_invalid_scaler(setup_values):
    with pytest.raises(ValueError):
        data = (0, 0, 0, 0)
        setup_values.scale_data_columns(data, "invalid_scaler")


# Test get_feature_and_class_count method
def test_get_feature_and_class_count(setup_values):
    assert setup_values.get_feature_and_class_count() == (3, 2)


# TEST DataValidator CLASS
def test_validate_data(setup_data):
    assert DataValidator.validate_data(setup_data.df, "target") is None


def test_validate_data_target(setup_data):
    assert (
        DataValidator.validate_target_column(setup_data.df, "target") is None
    )


def test_validate_data_invalid_target(setup_data):
    with pytest.raises(ValueError):
        DataValidator.validate_target_column(setup_data.df, "invalid_target")


def test_validate_data_drop_col(setup_data):
    assert (
        DataValidator.validate_drop_columns(setup_data.df, ["feature1"])
        is None
    )


def test_validate_data_invalid_drop_col(setup_data):
    with pytest.raises(ValueError):
        DataValidator.validate_drop_columns(setup_data.df, ["invalid_col"])


# TEST DataCleaner CLASS


# Test clean_data method
def test_clean_data_high_cardinality():
    high_cardinality_df = pd.DataFrame(
        {
            "feature1": list(map(str, range(16))),
            "feature2": np.random.choice(["A", "B"], 16),
            "feature3": np.random.randn(16),
            "target": np.random.randint(0, 2, 16),
        }
    )
    high_cardinality_df = DataCleaner.clean_data(high_cardinality_df)
    assert high_cardinality_df.shape[1] == 3


def test_clean_data_remove_nan():
    nan_df = pd.DataFrame(
        {
            "feature1": np.random.choice(["A", "B", None], 100),
            "feature2": list(np.random.randn(98)) + [None, None],
            "target": list(np.random.randint(0, 2, 98)) + [None, None],
        }
    )
    nan_df = DataCleaner.clean_data(nan_df)
    assert not np.isnan(nan_df).any()


@pytest.fixture
def setup_df():
    return pd.DataFrame(
        {
            "feature1": np.random.choice(["A", "B", "c"], 100),
            "feature2": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }
    )


def test_clean_data_one_hot(setup_df):
    one_hot_df = DataCleaner.clean_data(setup_df)
    assert one_hot_df.shape[1] == 5


# Test separate_features method
def test_separate_features(setup_df):
    _, num_idx, cat_idx = DataCleaner.separate_features(setup_df)
    assert len(num_idx) == 2
    assert len(cat_idx) == 1


if __name__ == "__main__":
    pytest.main([__file__])
