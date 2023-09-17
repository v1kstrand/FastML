"""
FastML Main Module
==================

This is the main module for the FastML project. It's designed to simplify
the model selection process. The module orchestrates the end-to-end 
workflow, from handling user input to saving the best model after grid 
search.

Functions
---------
- handle_user_input : Collects and validates user input.
- transform_data : Transforms the dataset based on user input.
- prepare_models_and_grid : Readies models and grid for grid search.
- perform_grid_search : Executes grid search on the models.
- save_best_model : Saves the best model based on evaluation metrics.

Main Function
-------------
- main() : Orchestrates the entire workflow.

Usage
-----
Run this module as the entry point to the FastML application.
"""

from module.main_util import (
    handle_user_input,
    perform_grid_search,
    prepare_models_and_grid,
    save_best_model,
    transform_data,
)


def main() -> None:
    """Main function for the program"""

    # Handle user input
    user_input = handle_user_input()

    # Transform data
    train_test_data, num_features, num_classes = transform_data(user_input)

    # Prepare models and grid
    models, grid_result_holder, evaluator = prepare_models_and_grid(
        user_input["task_type"],
        train_test_data[3],  # y_test
        num_features,
        num_classes,
    )

    # Perform grid search
    perform_grid_search(train_test_data, evaluator, grid_result_holder, models)

    # Save the best model
    save_best_model(grid_result_holder, user_input["model_name"])


if __name__ == "__main__":
    main()
