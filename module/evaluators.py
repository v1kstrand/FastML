from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_recall_fscore_support as report_score
from sklearn.metrics import r2_score


class EvaluatorInterface(ABC):
    """
    Interface for evaluating a model. Contains methods for calculating
    the metrics, for a given model type,

    Methods:
     - evaluate(predictions: np.ndarray, model_type: str) -> None:
     - calculate_metrics(predictions: np.ndarray) -> list[float] -> None:
     - reset_score() -> None:
    """

    @abstractmethod
    def __init__(self, y_test: np.ndarray, classifier: bool):
        """init"""

    @abstractmethod
    def evaluate(self, predictions: np.ndarray, model_type: str) -> None:
        """
        Evaluate the model and update the model_metrics dictionary.

        Args:
         - predictions (np.ndarray): The predicted labels.
             model_type (str): The type of the model being evaluated.
        """

    @abstractmethod
    def calculate_metrics(self, predictions: np.ndarray) -> list[float]:
        """
        Calculate the evaluation metrics based on the model type.

        Args:
         - predictions (np.ndarray): The predicted labels.

        Returns:
         - list[float]: A list of evaluation metrics, including
             precision, recall, f1, and accuracy.
        """

    @abstractmethod
    def reset_score(self) -> None:
        """Reset the score dictionary."""


class ClassifierEvaluator(EvaluatorInterface):
    """
    EvaluatorInterface for classifier models. Contains methods for
    calculating and displaying precision, recall, f1, and accuracy.

    Args:
     - y_test (np.ndarray): The true labels of the test data.
     - grid_results_callback (callable): A callback
         function that takes in the grid search results.

    Methods:
     - evaluate(predictions: np.ndarray, model_type: str) -> None:
     - calculate_metrics(predictions: np.ndarray) -> list[float]:
     - reset_score() -> None:
     - display_metrics(metrics: Union[list[str], np.ndarray]) -> None:
    """

    def __init__(self, y_test: np.ndarray, grid_results_callback: callable):
        self.y_test: np.ndarray = y_test
        self.grid_results_callback: callable = grid_results_callback
        self.score: dict[str, float] = {}

    def evaluate(self, predictions: np.ndarray, model_type: str) -> None:
        """
        Evaluate the model and update the model_metrics dictionary.

        Args:
         - predictions (np.ndarray): The predicted labels.
             model_type (str): The type of the model being evaluated.

        """
        self.calculate_metrics(predictions)
        self.display_score(predictions)
        self.grid_results_callback(self.score, model_type)
        self.reset_score()

    def calculate_metrics(self, predictions: np.ndarray) -> list[float]:
        """
        Calculate the evaluation metrics: precision, recall, f1, and
        accuracy.

        Args:
         - predictions (np.ndarray): The predicted labels.

        Returns:
         - list[float]: A list of evaluation metrics, including
             precision, recall, f1, and accuracy.
        """
        macro_scores = report_score(self.y_test, predictions, average="macro")
        self.score["precision"] = macro_scores[0]
        self.score["recall"] = macro_scores[1]
        self.score["f1"] = macro_scores[2]
        self.score["accuracy"] = accuracy_score(self.y_test, predictions)

    def reset_score(self) -> None:
        """Reset the score dictionary."""
        return self.score.clear()

    def display_score(self, predictions: np.ndarray) -> None:
        """
        Display the evaluation metrics using the classification_report.

        Args:
         - metrics (np.ndarray): The evaluation metrics.
        """
        print(classification_report(self.y_test, predictions))
        print(confusion_matrix(self.y_test, predictions))


class RegressorEvaluator(EvaluatorInterface):
    """
    EvaluatorInterface for regression models. Contains methods for
    calculating mean absolute error, root mean squared error, and
    r2 score.

    Args:
     - y_test (np.ndarray): The true labels of the test data.
     - grid_results_callback (callable): A callback
         function that takes in the grid search results.

    Methods:
     - evaluate(predictions: np.ndarray, model_type: str) -> None:
     - calculate_metrics(predictions: np.ndarray) -> list[float]:
     - reset_score() -> None:

    """

    def __init__(self, y_test: np.ndarray, grid_results_callback: callable):
        self.y_test: np.ndarray = y_test
        self.grid_results_callback: callable = grid_results_callback
        self.score: dict[str, float] = {}

    def evaluate(self, predictions: np.ndarray, model_type: str) -> None:
        """
        Evaluate the model and update the model_metrics dictionary.

        Args:
         - predictions (np.ndarray): The predicted labels.
             model_type (str): The type of the model being evaluated.

        """
        self.calculate_metrics(predictions)
        self.grid_results_callback(self.score, model_type)
        self.reset_score()

    def calculate_metrics(self, predictions: np.ndarray) -> None:
        """
        Calculate the evaluation metrics: precision, recall, f1, and

        Args:
         - predictions (np.ndarray): The predicted labels.

        Returns:
         - list[float]: A list of evaluation metrics, including
             precision, recall, f1, and accuracy.
        """
        self.score["mae"] = mean_absolute_error(self.y_test, predictions)
        self.score["rmse"] = mse(self.y_test, predictions) ** 0.5
        self.score["r2"] = r2_score(self.y_test, predictions)

    def reset_score(self) -> None:
        """Reset the score dictionary."""
        return self.score.clear()
