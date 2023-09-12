from collections import Counter

from sklearn.model_selection import GridSearchCV


class GridResult:
    """
    This class preserves the results of a grid search.
    It stores the evaluation scores and models in dictionaries.

    Args:
     - score_dict (dict[str, tuple[int, str, str]]): A dictionary
         containing the evaluation scores, models, and other information.
         Where {metric: (score, model, comparator)}. (comparator is
         either '>' or '<' depending on whether a higher or lower score
         is desired, such as for accuracy or loss.)

    Attributes:
     - grid_results (dict[str, GridSearchCV]): A dictionary to store
         the grid search results. e.g. {LogisticRegression: grid_search_object}

    Methods:
     - add_grid_result(grid: GridSearchCV, model_type: str) -> None:
     - update_score_dict(score: list[int], model_type: str) -> None:
     - evaluate_models_and_get_best() -> GridSearchCV:
     - display_results(best_model: str) -> None:
    """

    def __init__(self, score_dict: dict[str, tuple[int, str, str]]):
        self.score_dict: dict[str, tuple[int, str, str]] = score_dict
        self.grid_results: dict[str, GridSearchCV] = {}

    def add_grid_result(self, grid: GridSearchCV, model_type: str) -> None:
        """Add the grid result to the grid_results dictionary.

        Parameters:
         - grid (GridSearchCV): The grid search object containing the results.
             model_type (str): The type of the model.

        """
        self.grid_results[model_type] = grid

    def update_score_dict(self, score: list[int], model_type: str):
        """Update the score dictionary based on the evaluation.

        This method takes a list of scores and a model type as input
        parameters.
        It compares each score with the current highest score for
        each metric in the score dictionary.
        The comparators ('>' or '<') are used to determine whether a
        higher or lower score is desired for each metric.

        Parameters:
         - score (list[int]): A list of scores for each metric.
         - model_type (str): The type of the model.
        """
        comparison_func = {">": lambda x, y: x > y, "<": lambda x, y: x < y}

        for idx, metric in enumerate(self.score_dict.keys()):
            curr_high_score, _, comparator = self.score_dict[metric]
            curr_comparison = comparison_func[comparator]
            if curr_comparison(score[idx], curr_high_score):
                self.score_dict[metric] = (score[idx], model_type, comparator)

    def evaluate_models_and_get_best(self) -> GridSearchCV:
        """Get the best model based on the evaluation metrics.

        This method evaluates the models based on the evaluation
        metrics stored in the score dictionary.
        It checks each metric and counts the number of times a model
        has the highest score for that metric. The model thats
        appears the most is the best model.
        The `display_results` method is called to display the results
        of the best model.

        Returns:
         - GridSearchCV: The best model from the grid search results.

        """
        model_count = Counter(m for _, m, _ in self.score_dict.values())
        best_model = model_count.most_common(1)[0][0]
        self.display_results(best_model)
        return self.grid_results[best_model]

    def display_results(self, best_model: str) -> None:
        """Display the results of the grid search.

        This method displays the evaluation results stored in the
        score dictionary.
        It iterates over each metric in the score dictionary and prints
        the prints the top score and model for that metric.
        Finally, it prints a summary line indicating the best model
        for the dataset.

        Args:
         - best_model (str): The name of the best model.

        """
        for metric, (score, model, _) in self.score_dict.items():
            print(f"Best {metric} - score: {score:.2f} from {model}")
        print(f"\nRESULTS:\n{best_model} is the best model for this dataset")
