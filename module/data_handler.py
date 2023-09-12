from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from module.utils import read_data


class DataValidator:
    """
    Class for validating data in a DataFrame.

    Methods:
     - validate_data(self, df: pd.DataFrame, target: str, drop_col:
         Optional[list[str]] = None) -> None:
     - validate_target_column(self, df: pd.DataFrame, target: str)
         -> None:
     - validate_drop_columns(self, df: pd.DataFrame, drop_col:
         list[str]) -> None:
    """

    @staticmethod
    def validate_data(
        df: pd.DataFrame, target: str, drop_col: Optional[list[str]] = None
    ) -> None:
        """
        DataValidator main function for validating data.

        Args:
         - df (pd.DataFrame): The DataFrame to validate.
             target (str): The name of the target column.
         - drop_col (Optional[List[str]], optional): A list of column
             names to drop. Defaults to None.
        """
        drop_col = drop_col or []
        DataValidator.validate_target_column(df, target)
        DataValidator.validate_drop_columns(df, drop_col)

    @staticmethod
    def validate_target_column(df: pd.DataFrame, target: str) -> None:
        """
        Validate the target column in the DataFrame.

        Args:
         - df (pd.DataFrame): The DataFrame to validate.
             target (str): The name of the target column.
        """
        if target not in df.columns:
            raise ValueError(f"Target column: {target} not found in data")

    @staticmethod
    def validate_drop_columns(df: pd.DataFrame, drop_col: list[str]) -> None:
        """
        Validate the columns to drop in the DataFrame.

        Args:
         - df (pd.DataFrame): The DataFrame to validate.
         - drop_col (List[str]): A list of column names to drop.
        """
        if any(c not in df.columns for c in drop_col):
            raise ValueError(
                f"Columns to drop: column in {drop_col} not found in data"
            )


class DataCleaner:
    """
    Class for cleaning data

    Methods:
     - clean_data(X: pd.DataFrame) -> np.ndarray:
     - separate_features(X: pd.DataFrame) -> tuple[pd.DataFrame,
         list[str], list[str]]:
    """

    @staticmethod
    def clean_data(X: pd.DataFrame) -> np.ndarray:
        """
        Cleans the data by imputing missing values and one-hot encoding
        categorical columns.

        Args:
         - X (pd.DataFrame): The input dataframe containing the data to
             be cleaned.

        Returns:
         - np.ndarray: the cleaned data as a numpy array
        """
        X, numerical, categorical = DataCleaner.separate_features(X)
        num_pipeline = Pipeline([("impute", SimpleImputer(strategy="mean"))])
        cat_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "one-hot",
                    OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                ),
            ]
        )
        full_pipeline = ColumnTransformer(
            [
                ("num", num_pipeline, numerical),
                ("cat", cat_pipeline, categorical),
            ]
        )
        return full_pipeline.fit_transform(X)

    @staticmethod
    def separate_features(
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """
        Preparing the data for cleaning by separating the numerical
        and categorical features.

        Args:
         - X (pd.DataFrame): The input dataframe containing the data to
             be separated.

        Returns:
         - tuple[pd.DataFrame, list[str], list[str]]: A tuple
             containing (1) the cleaned dataframe,
             (2) a list of numerical feature names, (3) and a list of
             categorical feature names.
        """
        numerical = X.select_dtypes(include=np.number).columns.tolist()
        categorical = X.select_dtypes(exclude=np.number)
        categorical = categorical.loc[:, X.nunique() <= 15].columns.tolist()
        return X[numerical + categorical], numerical, categorical


class DataTransformer:
    """
    Class for transforming data into a format suitable for training
    and testing.

    Args:
     - source (Union[str, pd.DataFrame]): The data source.
         It can be either a file path or a DataFrame.
     - loader_func (callable, optional): A function to load the data
         from the source argument. Defaults to read_data.
     - cleaner_callback (callable, optional): A callback function
         to clean the data. Defaults to DataCleaner.clean_data.
     - validator_callback (callable, optional): A callback function
         to validate the data. Defaults to DataValidator.validate_data.

    Methods:
     - process_data(self, target: str, classifier: bool, split: float,
         drop_col: Optional[list[str]] = None) -> tuple[np.ndarray]:

     - set_x_y_data(self, target: str, drop_col: Optional[list[str]]
         = None) -> None:

     - set_correct_data_type_and_shape(self, classifier: bool) -> None:

     - split_data_into_train_test(self, split: float, classifier: bool)
         -> tuple[np.ndarray]:

     - scale_data_columns(self, data: tuple[np.ndarray], zscore: bool)
         -> tuple[np.ndarray]:

     - get_feature_and_class_count(self) -> tuple[int]:
    """

    def __init__(
        self,
        source: Union[str, pd.DataFrame],
        loader_func: callable = None,
        cleaner_callback: callable = None,
        validator_callback: callable = None,
    ) -> None:
        loader_func = loader_func or read_data
        self.df: pd.DataFrame = loader_func(source)
        self.X: Union[pd.DataFrame, np.ndarray] = None
        self.y: np.ndarray = None
        self.cleaner_callback: callable = (
            cleaner_callback or DataCleaner.clean_data
        )
        self.validator_callback: callable = (
            validator_callback or DataValidator.validate_data
        )

    def process_data(
        self,
        target: str,
        classifier: bool,
        split: float,
        scale: str = None,
        drop_col: Optional[list[str]] = None,
    ) -> tuple[np.ndarray]:
        """
        Orchestrates the data transformation process.

        Args:
         - target (str): The target variable.
             classifier (bool): Indicates whether the problem is a
         - classification problem or not.
         - split (float): The ratio to split the data into training
             and testing sets.
        - scale (str): The type of scaling to be applied. It can be
             either 'zscore', 'minmax' or None. Defaults to None for
             no scaling.
         - drop_col (Optional[List[str]], optional): A list of column
             names to be dropped from the data. Defaults to None.

        Returns:
         - tuple[np.ndarray]: A tuple containing the scaled training and
             testing data as (X_train, X_test, y_train, y_test)
        """
        self.validator_callback(self.df, target, drop_col)
        self.set_x_y_data(target, drop_col)
        self.X = self.cleaner_callback(self.X)
        self.set_correct_data_type_and_shape(classifier)
        data = self.split_data_into_train_test(split, classifier)
        return self.scale_data_columns(data, scale) if scale else data

    def set_x_y_data(
        self, target: str, drop_col: Optional[list[str]] = None
    ) -> None:
        """
        Setting the X and y data.

        Args:
         - target (str): The name of the target variable.
         - drop_col (Optional[list[str]], optional): A list of column
             names to be dropped from the dataframe. Defaults to None.
        """
        drop_col = drop_col or []
        self.X = self.df.drop(drop_col + [target], axis=1)
        self.y = self.df[target].values

    def set_correct_data_type_and_shape(self, classifier: bool) -> None:
        """
        Setting the correct data type and shape for the data.

        Args:
         - classifier (bool): A flag indicating whether the problem is
             a classification problem or not.
        """
        self.X = self.X.astype("float32")
        dtype, shape = ("int64", None) if classifier else ("float32", (-1, 1))
        self.y = self.y.astype(dtype).reshape(shape)

    def split_data_into_train_test(
        self, split: float, classifier: bool
    ) -> tuple[np.ndarray]:
        """
        Splits the data into train and test sets.

        Args:
         - split (float): The proportion of the data to be used for
             testing.
         - classifier (bool): A flag indicating whether the problem is a
             classification problem or not.

        Returns:
         - tuple[np.ndarray]: A tuple containing the train and test sets.
        """
        stratify = self.y if classifier else None
        return train_test_split(
            self.X, self.y, test_size=split, stratify=stratify, shuffle=True
        )

    @staticmethod
    def scale_data_columns(
        data: tuple[np.ndarray], scale: str
    ) -> tuple[np.ndarray]:
        """
        Scales the numerical columns in the data.

        Args:
         - data (tuple[np.ndarray]): A tuple containing the train and
             test sets.
         - scale (str): The type of scaling to be applied. It can be
                either 'zscore' or 'minmax'.

        Returns:
         - tuple[np.ndarray]: A tuple containing the scaled train
             and test sets.
        """
        if scale not in {"zscore", "minmax"}:
            raise ValueError("scale must be either 'zscore' or 'minmax' (str)")

        X_train, X_test, y_train, y_test = data

        scaler = StandardScaler() if scale == "zscore" else MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def get_feature_and_class_count(self) -> tuple[int]:
        """
        Returns the number of classes and features in the data.

        Returns:
         - tuple[int]: A tuple containing the number of features and the
             number of unique classes.
        """
        return self.X.shape[1], len(np.unique(self.y))
