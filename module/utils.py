import os
from math import inf
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator

# Maximum number of unique values in y for classification models
CLASSIFICATION_THRESHOLD = 50

# User file template
USER_FILE = (
    "Complete this form before running the application by filling in the \n"
    "details after each '->' sign.\n\n"
    "Example: ' task_type - type: regression / classification -> classification '\n\n"
    "# task_type - type: regression / classification ->\n\n"
    "# csv_path - Path or URL to CSV, example: C:/Users/.../data.csv ->\n\n"
    "# target - Target column name, example: target ->\n\n"
    "# split_size - Train test split size (0 <= x <= 1), example: 0.3 ->\n\n"
    "# model_name - Final model name, example: final_model ->\n\n"
    "# scale - (Optional) Type: zscore / minmax ->\n\n"
    "# drop_columns - (Optional) Columns to drop, separate with '|' example: col1|col2|col3 ->"
)


def get_user_choice(limit: int, msg: str = "", tries: int = inf) -> int:
    """
    Function that gets the user's input choice (number between 0 and
    limit) and validates and returns it.

    Args:
     - limit (int): The maximum choice number.
     - msg (str): The message to be printed to the user.
     - tries (int, optional): The maximum number of of choices the user
         can make. Defaults to inf.

    Returns:
     - int: The user's choice.

    Raises:
     - ValueError: If the user's choice is invalid after the maximum
         number of tries
    """
    choice = input(msg)
    while tries > 0 and (not choice.isdigit() or not 0 < int(choice) <= limit):
        print("Invalid choice.")
        choice = input(msg)
        tries -= 1

    if tries <= 0 and (not choice.isdigit() or not 0 < int(choice) <= limit):
        raise ValueError("Too many invalid choices.")

    return int(choice) - 1


class FileHandler:
    """Class that handles file related operations.

    Methods:
     - read(get_choice: callable = get_user_choice) -> str:
     - get_user_input_files() -> list[str]:
     - write_file() -> None:
    """

    user_folder = Path("user_input")
    user_file_path = os.path.join(user_folder, "user_input.txt")

    @staticmethod
    def read(get_choice: callable = get_user_choice) -> str:
        """
        Locates all the text files in the user_input folder and
        lets the user choose which file to open.

        Args:
         - get_choice (callable, optional): A function that gets the
             user's input choice. Defaults to get_choice.

        Returns:
         - str: The content of the chosen file.
        """
        text_files = FileHandler.get_user_input_files()
        FileHandler.print_files(text_files)

        msg = "Enter the number of the file you want to open: "
        choice = get_choice(len(text_files), msg)
        folder = FileHandler.user_folder
        chosen_file_path = os.path.join(folder, text_files[choice])

        with open(chosen_file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def print_files(text_files: list[str]) -> None:
        print("Available text files:")
        for idx, file in enumerate(text_files):
            print(f"{idx + 1}. {file}")

    @staticmethod
    def get_user_input_files(folder=None) -> list[str]:
        """
        Finds all the text files in the user_input folder, (or
        a specified folder) and returns a list of their names for the
        user to choose from.

        Args:
         - folder (str, optional): The path to the user_input folder.
             Defaults to None.

        Returns:
         - list[str]: A list of all the text files paths.
        """
        folder = folder or FileHandler.user_folder
        return FileHandler.validate_user_path(folder)

    @staticmethod
    def validate_user_path(folder: str) -> list[str]:
        """
        Validates that the user_input folder and that it contains
        at least one text file. If the folder does not exist,
        a new folder is created. If the folder is empty,
        it generates a new user_input file and raises a
        FileNotFoundError. Else, it returns a list
        of all the text files names in that folder.

        Args:
         - folder (str): The path to the user_input folder.

        Returns:
         - list[str]: A list of all the text files names.

        Raises:
         - FileNotFoundError: If the user_input folder is empty.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        if files := [f for f in os.listdir(folder) if f.endswith(".txt")]:
            return files
        FileHandler.write_file()
        raise FileNotFoundError(f"{folder} is empty, generating new file...")

    @staticmethod
    def write_file(file_path=None) -> None:
        """
        Writes a new user_file at the specified file to the FileHandler
        user_file_path path.

        """
        user_file_path = file_path or FileHandler.user_file_path
        with open(user_file_path, "w", encoding="utf-8") as f:
            f.write(USER_FILE)


def validate_data_type(
    classifier: bool, num_classes: int, strict: bool = True
) -> bool:
    """
    Checking if the target column is of the correct type for the model
    by checking if the number of unique values in y is less than
    CLASSIFICATION_THRESHOLD.

    Args:
     - classifier (bool): Indicates whether the model is a
         classification model (True) or a regression model (False).
         num_classes (int): The number of classes for classification
         models.
     - strict (bool, optional): Determines whether to raise an error
         if the target type does not match the model type. Defaults to
         True.

    Returns:
        bool: True if the target type matches the model type, False
          otherwise.

    Raises:
        ValueError: If strict is True and the target type does not
          match the model type.
    """

    model_type = "classification" if classifier else "regression"
    target_type = (
        "classification"
        if num_classes <= CLASSIFICATION_THRESHOLD
        else "regression"
    )

    if strict and target_type != model_type:
        raise ValueError(
            f"Target column indicates {target_type}, "
            f"your model is a {model_type} model"
        )
    return target_type == model_type


def print_headline(msg: str) -> None:
    """Prints a headline with a message in the middle.

    Args:
        - msg (str): The message to be printed in the the headline.
    """

    print(f"\n{'- '*30}\n{msg}\n{'- '*30}")


def save_model(model: BaseEstimator, name: str, model_type: str) -> None:
    """
    Saving the model as a joblib file.

    Args:
     - model (BaseEstimator): The model object to be saved.
     - name (str): The name of the saved model file.
     - path (str, optional): The path where the model file
         should be saved. Defaults to "".
    """
    os.makedirs("models", exist_ok=True)

    print(f"\nSaving model as '{name}_{model_type}.joblib'")
    full_path = os.path.join("models", f"{name}_{model_type}.joblib")
    dump(model, full_path)
    print_headline("PLEASE ENJOY YOUR NEW MODEL")


def read_data(data) -> pd.DataFrame:
    """
    Validating the data by checking if it's a path to a csv file
    or a pandas DataFrame.
    If it's a csv file, it reads the file using pandas.read_csv().
    If it's a DataFrame, it returns the DataFrame as is.
    Otherwise, it raises a FileNotFoundError or TypeError.

    Args:
     - data: The input data, which can be either a path to a csv
         file or a pandas DataFrame.

    Returns:
     - pd.DataFrame: The validated data as a pandas DataFrame.

    Raises:
     - FileNotFoundError: If the provided data is a path to a csv
         file that does not exist.
     - TypeError: If the provided data is neither a path to a csv
         file nor a pandas DataFrame.
    """
    if isinstance(data, str):
        try:
            return pd.read_csv(data)
        except FileNotFoundError as exp:
            raise FileNotFoundError(f"File '{data}' not found") from exp
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise TypeError(
            "Data must be a path to a csv file or a pandas DataFrame"
        )
