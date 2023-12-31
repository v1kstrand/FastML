"""
FastML Grid Result Module
=========================

This module is designed to store and manage the results of grid search
operations. It holds metrics and configurations for each model, allowing
for easy comparison and selection of the best model.

Classes
-------
- GridResult : Class for storing grid search results and metrics.

Usage
-----
Import this module to store and manage grid search results in your FastML
workflow.
"""

from collections import defaultdict

from sklearn.model_selection import GridSearchCV

from module.utils import get_user_choice


class GridResult:
    """
    This class preserves the results of a grid search.
    It stores the evaluation scores and models in dictionaries.

    Args:
     - metrics (list[str]): A list of all the metrics being evaluated

    Attributes:
     - grid_results (dict[str, GridSearchCV]): A dictionary to store
         the grid search results. e.g. {"SVC": grid_search_object}
     - score_dict (dict[str, dict[str, int]]): A dictionary to store
         the score for each model based on the metrics in self.metricss

    Methods:
     - add_grid_result(grid: GridSearchCV, model_type: str) -> None:
     - update_score_dict(score: list[int], model_type: str) -> None:
     - let_user_pick_preferred_model() -> tuple[GridSearchCV, str]:
     - display_results() -> None:
     - display_suggested_model() -> None:
    """

    def __init__(self, metrics: list[str]):
        self.metrics: list[str] = metrics
        self.score_dict: dict[str, dict[str, int]] = defaultdict(dict)
        self.grid_results: dict[str, GridSearchCV] = {}

    def add_grid_result(self, grid: GridSearchCV, model_type: str) -> None:
        """Add the grid result to the grid_results dictionary.

        Parameters:
         - grid (GridSearchCV): The grid search object containing the results.
             model_type (str): The type of the model.

        """
        self.grid_results[model_type] = grid

    def update_score_dict(self, score: dict[str, int], model_type: str):
        """Update the score dictionary based on the scores

        Parameters:
         - score (dict[str, int]): A dict of scores for each metric.
         - model_type (str): The type of the model.
        """
        for metric in self.metrics:
            self.score_dict[model_type][metric] = score[metric]

    def let_user_pick_preferred_model(
        self, get_choise: callable = get_user_choice
    ) -> tuple[GridSearchCV, str]:
        """Let the user pick the preferred model from the grid search
        results. A suggestion is provided based on the scores.
        Finally, the selected model and a string of
        the model type is returned.

        Returns:
         - GridSearchCV: The best model from the grid search results.
         - model (str): The type of the model.

        """
        self.display_results()
        self.display_suggested_model()
        msg = "Please select the model you prefer: "
        selection = get_choise(len(self.score_dict), msg)
        model = list(self.score_dict.keys())[selection]
        return self.grid_results[model], model

    def display_results(self) -> None:
        """Display the results of the grid search."""
        print("\n\nEVALUATION RESULTS:\n")
        for i, (model, metrics) in enumerate(self.score_dict.items(), 1):
            model_metrics = "".join(
                f"{metric}: {value:.2f}, " for metric, value in metrics.items()
            )
            print(f"{i}. {model.ljust(20)} - {model_metrics[:-2]}")

    def display_suggested_model(self) -> None:
        """Suggest the best model based on the evaluation metrics.
        For classification, the calculation is based on highest added
        accuracy and f1.
        For regression, the calculation is based on highest r2."""

        best_model = defaultdict(int)
        for model in self.score_dict:
            best_model[model] += self.score_dict[model].get("accuracy", 0)
            best_model[model] += self.score_dict[model].get("f1", 0)
            best_model[model] += self.score_dict[model].get("r2", 0)
        print(f"\nFastMl suggests: {max(best_model, key=best_model.get)}")
