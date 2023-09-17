"""
FastML Grid Factory Module
==========================

This module is responsible for preparing the grid search configurations
and evaluators for machine learning models. It serves as a factory for
creating grid search setups tailored for classification and regression.

Classes
-------
- GridFactory : Factory class for grid search configurations.

Usage
-----
Import this module to generate grid search configurations and evaluators
for use in your FastML workflow.
"""

import numpy as np

from module.evaluators import (
    ClassifierEvaluator,
    EvaluatorInterface,
    RegressorEvaluator,
)
from module.grid_result import GridResult


class GridFactory:
    """
    Factory class for building grid result and evaluator objects.

    This class provides methods for building grid result and evaluator
    objects based on different types of models.

    The purpose of this class is to encapsulate the logic for creating
    these objects and provide a centralized place for managing their
    creation.

    Methods:
     - build(classifier: bool, y_test: np.ndarray) -> tuple[GridResult,
         EvaluatorInterface]:
     - regressor_build(y_test) -> tuple[GridResult, RegressorEvaluator]:
     - classifier_build(y_test) -> tuple[GridResult, ClassifierEvaluator]:
    """

    @staticmethod
    def build(
        classifier: bool, y_test: np.ndarray
    ) -> tuple[GridResult, EvaluatorInterface]:
        """
        Build the grid result and evaluator for a model depending on
        the task type, where the task type is either classification
        or regression.

        Args:
         - classifier (bool): A boolean flag indicating whether the task
             is classification (True) or regression (False).
        - y_test (np.ndarray): The true labels of the test dataset.

        Returns:
         - tuple[GridResult, EvaluatorInterface]: A tuple containing
             the `grid_result` and `evaluator` objects.
        """
        if classifier:
            return GridFactory.classifier_build(y_test)
        return GridFactory.regressor_build(y_test)

    @staticmethod
    def regressor_build(y_test) -> tuple[GridResult, RegressorEvaluator]:
        """
        Builds the grid result and evaluator for a regression models.

        Creates a grid_result object by giving it a list of regression
        metrics as input (mae, rmse, and r2).

        Creates an evaluator object by passing the `y_test` variable
        and the `grid_result.update_score_dict` method as arguments.
        grid_result.update_score_dict is a method that let the evaluator
        update the score dictionary of the grid_result object.

        Args:
         - y_test: The true labels of the test dataset.

        Returns:
         - tuple[GridResult, RegressorEvaluator]: A tuple containing
             the `grid_result` and `evaluator` objects.
        """
        metrics = ["mae", "rmse", "r2"]
        grid_result = GridResult(metrics)
        evaluator = RegressorEvaluator(y_test, grid_result.update_score_dict)
        return grid_result, evaluator

    @staticmethod
    def classifier_build(y_test) -> tuple[GridResult, ClassifierEvaluator]:
        """
        Builds the grid result and evaluator for a classification
        models.

        Creates a grid_result object by giving it a list of
        classification metrics as input (precision, recall, f1, and
        accuracy).

        Creates an evaluator object by passing the `y_test` variable
        and the `grid_result.update_score_dict` method as arguments.
        grid_result.update_score_dict is a method that let the evaluator
        update the score dictionary of the grid_result object.

        Args:
         - y_test: The true labels of the test dataset.

        Returns:
         - tuple[GridResult, ClassifierEvaluator]: A tuple containing
             the `grid_result` and `evaluator` objects.
        """
        metrics = ["precision", "recall", "f1", "accuracy"]
        grid_result = GridResult(metrics)
        evaluator = ClassifierEvaluator(y_test, grid_result.update_score_dict)
        return grid_result, evaluator
