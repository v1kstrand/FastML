import numpy as np
from sklearn.base import BaseEstimator

from module.data_handler import DataTransformer
from module.evaluators import EvaluatorInterface
from module.grid_factory import GridFactory
from module.grid_result import GridResult
from module.grid_search import GridSearch
from module.model_factory import HyperParams, ModelFactory
from module.user_input_handler import UserInput, UserInputHandler
from module.utils import save_model, validate_data_type


def handle_user_input() -> UserInput:
    """
    Handles user input and returns a UserInput object.

    Args:
     - path (str): The path to the user input file.

    Returns:
     - UserInput: The processed user input.
    """
    user_handler = UserInputHandler()
    return user_handler.process()


def transform_data(user_input: UserInput) -> tuple[np.ndarray]:
    """
    Transforms user input data and returns the processed data,
    number of features, and number of classes.

    Args:
     - user_input (UserInput): A dictionary containing user
         input data.

    Returns:
     - tuple[np.ndarray]: A tuple containing the processed data,
         number of features, and number of classes.
    """
    data_transformer = DataTransformer(user_input["csv_path"])
    train_test_data = data_transformer.process_data(
        user_input["target"],
        user_input["task_type"],
        user_input["split_size"],
        user_input["scale"],
        user_input["drop_columns"],
    )
    num_features, num_classes = data_transformer.get_feature_and_class_count()
    validate_data_type(user_input["task_type"], num_classes)
    return train_test_data, num_features, num_classes


def prepare_models_and_grid(
    task_type: bool,
    y_test: np.ndarray,
    num_features: int,
    num_classes: int,
) -> tuple[dict[BaseEstimator, HyperParams], GridResult, EvaluatorInterface]:
    """
    Prepares models and params, grid result object, and evaluator.

    Args:
     - task_type (bool): A boolean indicating the type of task.
         y_test (np.ndarray): An array containing the test labels.
     - num_features (int): The number of features in the dataset.
         num_classes (int): The number of classes in the dataset.

    Returns:
     - tuple[dict[BaseEstimator, HyperParams], GridResult,
         EvaluatorInterface]:
     - A tuple containing the models, GridResult object, and
         evaluator object.
    """
    models = ModelFactory.build(task_type, num_features, num_classes)
    grid_result, evaluator = GridFactory.build(task_type, y_test)
    return models, grid_result, evaluator


def perform_grid_search(
    train_test_data: tuple[np.ndarray],
    evaluator: callable,
    grid_result: callable,
    models: dict[BaseEstimator, HyperParams],
) -> None:
    """
    Performs grid search using the provided train-test data,
    evaluator, grid result, and models.

    Args:
     - train_test_data (tuple[np.ndarray]): A tuple containing the
         train and test data.
     - evaluator (callable): A callable object for evaluating the models.
     - grid_result (callable): A callable object for storing the
         grid search results.
     - models (dict[BaseEstimator, HyperParams]): A dictionary
         containing the models and their hyperparameters.
    """
    evaluator_callback = evaluator.evaluate
    grid_result_callback = grid_result.add_grid_result
    grid_search = GridSearch(
        train_test_data, evaluator_callback, grid_result_callback, models
    )
    grid_search.execute()


def save_best_model(grid_result: GridResult, model_name: str) -> None:
    """
    Saves the best model from the grid search results.

    Args:
     - grid_result (GridResult): The grid search results.
     - model_name (str): The name of the model to be saved.

    """
    final_model = grid_result.evaluate_models_and_get_best()
    save_model(final_model, model_name)
