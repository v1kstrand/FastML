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
        Builds the grid result and evaluator for a regression model.

        This function takes the `y_test` variable as input, which
        represents the true labels of the test dataset.
        It initializes a `score_dict` dictionary to store the initial
        score values for mae, rmse, and r2.
        The scores are initialized with negative infinity
        and infinity values (depending on the metric type),
        empty strings for best best model type, and ">" as the
        comparison operator. (e.g. for mae, the score is initialized
        with (np.inf, "", "<") because a lower score is desired.
        For r2, the score is initialized with (-np.inf, "", ">")
        because a higher score is desired.)

        A `grid_result` object is created using the `score_dict`
        dictionary, which will be used to store and update the scores
        during the grid search process.

        An `evaluator` object is also created by passing the `y_test`
        variable and the `grid_result.update_score_dict` method as
        arguments.

        The `evaluator` object is responsible for evaluating the
        performance of the regression models and updating the
        `grid_result` object with the new scores.

        Finally, the function returns a tuple containing the
        `grid_result` and `evaluator` objects.

        Args:
         - y_test: The true labels of the test dataset.

        Returns:
         - tuple[GridResult, RegressorEvaluator]: A tuple containing
             the `grid_result` and `evaluator` objects.
        """
        score_dict = score_dict = {
            "mae": (np.inf, "", "<"),
            "rmse": (np.inf, "", "<"),
            "r2": (-np.inf, "", ">"),
        }
        grid_result = GridResult(score_dict)
        evaluator = RegressorEvaluator(y_test, grid_result.update_score_dict)
        return grid_result, evaluator

    @staticmethod
    def classifier_build(y_test) -> tuple[GridResult, ClassifierEvaluator]:
        """
        Builds the grid result and evaluator for a classifier model.

        This function takes the `y_test` variable as input, which
        represents the true labels of the test dataset.
        It initializes a `score_dict` dictionary to store the initial
        score values for precision, recall, f1, and accuracy.
        The scores are initialized with negative infinity values,
        empty strings for best best model type, and ">" as the
        comparison operator.

        A `grid_result` object is created using the `score_dict`
        dictionary, which will be used to store and update the scores
        during the grid search process.

        An `evaluator` object is also created by passing the `y_test`
        variable and the `grid_result.update_score_dict` method as
        arguments.

        The `evaluator` object is responsible for evaluating the
        performance of the classifier models and updating the
        `grid_result` object with the new scores.

        Finally, the function returns a tuple containing the
        `grid_result` and `evaluator` objects.

        Args:
         - y_test: The true labels of the test dataset.

        Returns:
         - tuple[GridResult, ClassifierEvaluator]: A tuple containing
             the `grid_result` and `evaluator` objects.
        """
        score_dict = {
            "precision": (-np.inf, "", ">"),
            "recall": (-np.inf, "", ">"),
            "f1": (-np.inf, "", ">"),
            "accuracy": (-np.inf, "", ">"),
        }
        grid_result = GridResult(score_dict)
        evaluator = ClassifierEvaluator(y_test, grid_result.update_score_dict)
        return grid_result, evaluator
