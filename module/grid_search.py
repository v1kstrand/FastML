"""
Grid Search Module for FastML
=============================

This module provides the GridSearch class, responsible for executing
grid search algorithms for hyperparameter tuning.

Classes
-------
- GridSearch : Orchestrates the grid search process.

Usage
-----
Import this module to utilize the GridSearch class for hyperparameter
tuning in machine learning tasks.

"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import GridSearchCV

from module.model_factory import HyperParams
from module.utils import print_headline

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


class GridSearch:
    """
    This class represents a grid search algorithm for hyperparameter
    tuning. It takes the following inputs:

    Args:
     - data (tuple[np.ndarray]): A tuple containing the training and
         testing data arrays in the order (X_train, X_test, y_train, _).

     - evaluator_callback (callable): A callback function that
         evaluates the predictions.

     - grid_results_callback (callable): A callback function that
         updates the grid results dictionary.

     - models (dict[BaseEstimator, HyperParams]): A dictionary mapping
         base estimator models to their corresponding hyperparameters.

      Methods:
      - execute() -> None:
      - process_model(model: BaseEstimator, model_type: str) -> None:
      - perform_grid_search(model: BaseEstimator) -> tuple[GridSearchCV,
          np.ndarray]:

    """

    def __init__(
        self,
        data: tuple[np.ndarray],
        evaluator_callback: callable,
        grid_results_callback: callable,
        models: dict[BaseEstimator, HyperParams],
    ):
        X_train, X_test, y_train, _ = data
        self.X_train: np.ndarray = X_train
        self.X_test: np.ndarray = X_test
        self.y_train: np.ndarray = y_train
        self.evaluator_callback: callable = evaluator_callback
        self.grid_results_callback: callable = grid_results_callback
        self.models: dict[BaseEstimator, HyperParams] = models

    def execute(self) -> None:
        """Main execution method for the grid search.

        Iterates over the models provided, gets each model name in
        yhr model_type variable as a string, (e.g "LogisticRegression")
        and processes the model using the `process_model` method.
        """
        for model in self.models:
            model_type: str = model.__class__.__name__
            self.process_model(model, model_type)

    def process_model(self, model: BaseEstimator, model_type: str) -> None:
        """Process the model and update the grid_results dictionary.

        This method processes the provided `model` of type
        `BaseEstimator` and updates the `grid_results` dictionary.
        It performs the following steps:

        1. Prints a headline for the `model_type`.
        2. Calls the `perform_grid_search` method to perform grid search
           using the `model` and obtain the grid search object `grid`
           and predictions `preds`.
        3. Calls the `evaluator_callback` method with `preds` and
           `model_type` as arguments to get the evaluation scores.
        4. Calls the `grid_results_callback` method with `grid` and
           `model_type` as arguments to save the grid search results.

        Args:
         - model (BaseEstimator): The base estimator model to be processed.
         - model_type (str): The type of the model.

        """
        print_headline(model_type)
        grid, preds = self.perform_grid_search(model)
        self.evaluator_callback(preds, model_type)
        self.grid_results_callback(grid, model_type)

    def perform_grid_search(
        self, model: BaseEstimator
    ) -> tuple[GridSearchCV, np.ndarray]:
        """Perform grid search and return results and predictions.

        This method performs a grid search using the provided `model`
        and the hyperparameters specified in `self.models[model]`.
        It fits the grid search on the training data (`self.X_train`
        and `self.y_train`) and prints the best parameters found.
        Finally, it returns the grid search object and the predictions
        made on the test data (`self.X_test`).

        Args:
         - model (BaseEstimator): The base estimator model to be used
             for grid search.

        Returns:
         - tuple[GridSearchCV, np.ndarray]: A tuple containing the grid
             search object and the predictions made on the test data.

        """
        grid = GridSearchCV(model, self.models[model], verbose=1, cv=5)
        grid.fit(self.X_train, self.y_train)
        print(f"Best params: {grid.best_params_}")
        return grid, grid.predict(self.X_test)
