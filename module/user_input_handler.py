import os
import re
from typing import Union

from module.utils import FileHandler

UserInput = dict[str, Union[str, bool, int, list[str]]]


class InputValidator:
    """
    Validates user input based on set of rules.

    Attributes:
    - validation_rules (dict[str, callable]): A dictionary containing
        validation rules for each field in the user file.

    Methods:
    - validate(user_input: dict[str, str]) -> None:
    - validate_keys(user_input: dict[str, str]) -> None:
    - validate_values(user_input: dict[str, str]) -> None:
    - validate_rules(user_input: dict[str, str]) -> None:
    """

    validation_rules = {
        "task_type": lambda x: x.lower() in ("classification", "regression"),
        "split_size": lambda x: x.replace(".", "", 1).isdigit()
        and 0 <= float(x) <= 1,
        "csv_path": lambda x: os.path.isfile(x),
        "target": None,
        "scale": lambda x: x.lower() in ("zscore", "minmax", "none"),
        "model_name": None,
        "drop_columns": None,
    }

    @staticmethod
    def validate(user_input: dict[str, str]) -> None:
        """
        Validates user input based on certain rules.

        Args:
         - user_input (dict[str, str]): A dictionary containing user
             input.

        """
        print(f"User input: {user_input}")
        InputValidator.validate_keys(user_input)
        InputValidator.validate_values(user_input)
        InputValidator.validate_rules(user_input)

    @staticmethod
    def validate_keys(user_input: dict[str, str]) -> None:
        """
        Validates that all required keys are present in the user input.

        Args:
         - user_input (dict[str, str]): A dictionary containing user
             input.

        Raises:
         - ValueError: If there is a mismatch in keys between the user
            input and the validation rules.
        """
        if set(InputValidator.validation_rules.keys()) != set(
            user_input.keys()
        ):
            raise ValueError(
                "Mismatch in keys. Please generate a new file by running"
                "the application without a user input file, "
                "and fill in all the fields."
            )

    @staticmethod
    def validate_values(user_input: dict[str, str]) -> None:
        """
        Validates that all required fields are filled in the user input.

        Args:
         - user_input (dict[str, str]): A dictionary containing user input.

        Raises:
         - ValueError: If any required field is not filled.
        """
        if not all(user_input[k] for k in user_input if k != "drop_columns"):
            raise ValueError(
                "Please fill in all the fields in the user input file. "
            )

    @staticmethod
    def validate_rules(user_input: dict[str, str]) -> None:
        """
        Validates each field in the user input according to its rule.

        Args:
         - user_input (dict[str, str]): A dictionary containing user input.

        Raises:
         - ValueError: If any field fails to meet its rule.
        """
        for key, rule in InputValidator.validation_rules.items():
            if rule and not rule(user_input[key]):
                raise ValueError(
                    f"Invalid value for {key}. Please check the user input file."
                )


class UserInputHandler:
    """
    Class for handling the input from the user file.

    Args:
    - filehandler_callback (callable): A callback function used for
        reading the user input file. If not provided, the default
        FileHandler.read function will be used.

    - validator_callback (callable): A callback function used for
        input validation. If not provided, the default
        InputValidator.validate function will be used.

    Attributes:
     - user_input (UserInput): A dictionary containing the user input.

    Methods:
    - process() -> UserInput:
    - extract_user_input(raw_input: str) -> dict[str, str]:
    - transform_user_input() -> None:
    """

    def __init__(
        self,
        filehandler_callback: callable = None,
        validator_callback: callable = None,
    ) -> None:
        self.user_input: UserInput = None
        self.filehandler_callback = filehandler_callback or FileHandler.read
        self.validator_callback: callable = (
            validator_callback or InputValidator.validate
        )

    def process(self) -> UserInput:
        """
        Main method for the UserInputHandler class.

        This method processes the user input by performing the following steps:
        1. Reads the raw input from a file specified by `file_path`.
        2. Extracts the user input from the raw input.
        3. Validates the user input using the `validator_callback` function.
        4. Transforms the user input into the desired format.

        Returns:
         - UserInput: The processed user input.

        Raises:
         - Any exceptions raised during the validation step.
        """
        raw_input = self.filehandler_callback()
        self.extract_user_input(raw_input)
        self.validator_callback(self.user_input)
        self.transform_user_input()
        return self.user_input

    def extract_user_input(self, raw_input: str) -> dict[str, str]:
        """
        Extracts the user input from the raw input and returns it as
        a dictionary.

        This method takes the `raw_input` string as input and extracts
        the user input
        by searching for lines that match the pattern:
        "# <key> - ... -> <value>".

        Args:
         - raw_input (str): The raw input string containing the user
            input.

        Returns:
         - dict[str, str]: A dictionary representing the extracted
            user input.
        """
        pattern = r"#\s*(\w+)\s*-\s*.*->\s*(.*)\s*"
        match = dict(re.findall(pattern, raw_input))
        self.user_input = {key: value.strip() for key, value in match.items()}

    def transform_user_input(self) -> None:
        """
        Transforms the user input stored in `self.user_input` dictionary.

        This method takes the user input stored in the
        `self.user_input` dictionary
        and performs transformations on specific keys to ensure
        compatibility with the rest of the code.

        """
        self.user_input["task_type"] = (
            self.user_input["task_type"].lower() == "classification"
        )
        self.user_input["split_size"] = float(self.user_input["split_size"])

        self.user_input["drop_columns"] = (
            self.user_input["drop_columns"].split("|")
            if self.user_input["drop_columns"]
            else []
        )
