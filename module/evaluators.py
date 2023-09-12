from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.metrics import precision_recall_fscore_support as report_score
from sklearn.metrics import r2_score


class EvaluatorInterface(ABC):
    """
    This is an abstract class for evaluating a model. It contains
    blueprint methods for evaluating a model, calculating the metrics,
    and displaying the metrics.

    Methods:
     - evaluate(predictions: np.ndarray, model_type: str) -> None:
     - calculate_metrics(predictions: np.ndarray) -> list[float] -> None:
     - display_metrics(metrics: Union[list[str], np.ndarray]) -> None:
    """

    @abstractmethod
    def __init__(self, y_test: np.ndarray, classifier: bool):
        pass

    @abstractmethod
    def evaluate(self, predictions: np.ndarray, model_type: str) -> None:
        """
        Evaluate the model and update the model_metrics dictionary.

        Args:
         - predictions (np.ndarray): The predicted labels.
             model_type (str): The type of the model being evaluated.
        """
        pass

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
        pass

    @abstractmethod
    def display_metrics(self, metrics: Union[list[str], np.ndarray]) -> None:
        """
        Display the evaluation metrics based on the model type.

        Args:
         - metrics (np.ndarray): The evaluation metrics.
        """
        pass


class ClassifierEvaluator(EvaluatorInterface):
    """
    This class implements the EvaluatorInterface amd is for
    evaluating a classifier model. Its main purpose is calculating the
    precision, recall, f1, and accuracy after a grid search.

    Args:
     - y_test (np.ndarray): The true labels of the test data.
     - grid_results_callback (callable): A callback
         function that takes in the grid search results.

    Methods:
     - evaluate(predictions: np.ndarray, model_type: str) -> None:
     - calculate_metrics(predictions: np.ndarray) -> list[float]:
     - display_metrics(metrics: Union[list[str], np.ndarray]) -> None:
    """

    def __init__(self, y_test: np.ndarray, grid_results_callback: callable):
        self.y_test: np.ndarray = y_test
        self.grid_results_callback: callable = grid_results_callback

    def evaluate(self, predictions: np.ndarray, model_type: str) -> None:
        """
        Evaluate the model and update the model_metrics dictionary.

        Args:
         - predictions (np.ndarray): The predicted labels.
             model_type (str): The type of the model being evaluated.

        """
        score = self.calculate_metrics(predictions)
        self.display_metrics(predictions)
        self.grid_results_callback(score, model_type)

    def calculate_metrics(self, predictions: np.ndarray) -> list[float]:
        """
        Calculate the evaluation metrics: precision, recall, f1, and

        Args:
         - predictions (np.ndarray): The predicted labels.

        Returns:
         - list[float]: A list of evaluation metrics, including
             precision, recall, f1, and accuracy.
        """
        macro_scores = report_score(self.y_test, predictions, average="macro")
        acc_score = accuracy_score(self.y_test, predictions)
        return list(macro_scores[:3]) + [acc_score]

    def display_metrics(self, metrics: np.ndarray) -> None:
        """
        Display the evaluation metrics using the classification_report.

        Args:
         - metrics (np.ndarray): The evaluation metrics.
        """
        print(classification_report(self.y_test, metrics))


class RegressorEvaluator(EvaluatorInterface):
    """
    This class implements the EvaluatorInterface amd is for
    evaluating a regressor model. Its main purpose is calculating the
    mean absolute error, root mean squared error, and r2 score after a
    grid search.


    Args:
     - y_test (np.ndarray): The true labels of the test data.
     - grid_results_callback (callable): A callback
         function that takes in the grid search results.

    Methods:
     - evaluate(predictions: np.ndarray, model_type: str) -> None:
     - calculate_metrics(predictions: np.ndarray) -> list[float]:
     - display_metrics(metrics: Union[list[str], np.ndarray]) -> None:
    """

    def __init__(self, y_test: np.ndarray, grid_results_callback: callable):
        self.y_test: np.ndarray = y_test
        self.grid_results_callback: callable = grid_results_callback

    def evaluate(self, predictions: np.ndarray, model_type: str) -> None:
        """
        Evaluate the model and update the model_metrics dictionary.

        Args:
         - predictions (np.ndarray): The predicted labels.
             model_type (str): The type of the model being evaluated.

        """
        score = self.calculate_metrics(predictions)
        self.display_metrics(score)
        self.grid_results_callback(score, model_type)

    def calculate_metrics(self, predictions: np.ndarray) -> list[float]:
        """
        Calculate the evaluation metrics: precision, recall, f1, and

        Args:
         - predictions (np.ndarray): The predicted labels.

        Returns:
         - list[float]: A list of evaluation metrics, including
             precision, recall, f1, and accuracy.
        """
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = r2_score(self.y_test, predictions)
        return [mae, rmse, r2]

    def display_metrics(self, metrics: list[str]) -> None:
        """
        Display the evaluation metric scores.

        Args:
         - metrics (np.ndarray): The evaluation metrics.
        """
        mae, rmse, r2 = metrics
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}\n")
