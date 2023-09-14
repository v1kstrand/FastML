from typing import Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from skorch import NeuralNetClassifier, NeuralNetRegressor

# Custom type hint for hyperparameters dictionary
HyperParams = dict[str, Union[int, float, str, torch.optim.Adam]]


class ModelFactory:
    """
    Factory class for creating models and their corresponding
    hyperparameters.

    Methods:
     - build(classifier: bool, num_features: int, num_classes: int)
         -> dict[BaseEstimator, HyperParams]:
     - regressor_build(num_features: int) -> dict[BaseEstimator,
         HyperParams]:
     - classifier_build(num_features: int, num_classes: int) ->
         dict[BaseEstimator, HyperParams]:
    """

    @staticmethod
    def build(
        classifier: bool, num_features: int, num_classes: int
    ) -> dict[BaseEstimator, HyperParams]:
        """
        Gets the model and its parameters according to the data type.

        Args:
         - classifier (bool): A boolean flag indicating whether the task
             is classification or regression.
         - num_features (int): The number of input features.
         - num_classes (int): The number of classes in the classification
             problem.

        Returns:
         - dict[BaseEstimator, HyperParams]: A dictionary containing
             the model as key and its corresponding hyperparameters as
             value.
        """
        if classifier:
            return ModelFactory.classifier_build(num_features, num_classes)
        return ModelFactory.regressor_build(num_features)

    @staticmethod
    def regressor_build(num_features: int) -> dict[BaseEstimator, HyperParams]:
        """
        Loading all the regression models, their parameters,
        in a dictionary for keeping track of the best model
        for each metric.

        Args:
         - num_features (int): The number of input features.

        Returns:
         - dict[BaseEstimator, HyperParams]: A dictionary containing the
             regression models as keys and their corresponding
             hyperparameters as values.
        """

        return {
            LinearRegression(): {},
            Lasso(): {"alpha": [0.02, 0.024, 0.025, 0.026, 0.03]},
            Ridge(): {"alpha": [200, 250, 270, 300, 500]},
            ElasticNet(): {
                "alpha": [0.1, 0.5, 1],
                "l1_ratio": [0.1, 0.5, 0.9],
            },
            SVR(): {"C": [1, 10, 100], "gamma": ["auto"]},
            NeuralNetRegressor(ANNRegressor): {
                "module__num_features": [num_features],
                "max_epochs": [60],
                "lr": [0.05],
                "optimizer": [torch.optim.Adam],
                "verbose": [0],
                "device": ["cuda" if torch.cuda.is_available() else "cpu"],
            },
        }

    @staticmethod
    def classifier_build(
        num_features: int, num_classes: int
    ) -> dict[BaseEstimator, HyperParams]:
        """
        Loading all the classification models, their parameters,
        in a dictionary for keeping track of the best model
        for each metric.

        Args:
         - num_features (int): The number of input features.
         - num_classes (int): The number of classes in the
              classification problem.

        Returns:
         - dict[BaseEstimator, HyperParams]: A dictionary containing the
             classification models as keys and their corresponding
             hyperparameters as values.

        """
        return {
            LogisticRegression(): {
                "C": np.logspace(-3, 3, 7),
                "max_iter": [1000],
            },
            KNeighborsClassifier(): {
                "n_neighbors": range(2, 7),
                "metric": ["euclidean", "manhattan"],
            },
            SVC(): {"C": [0.01, 1, 100], "gamma": ["scale", "auto"]},
            NeuralNetClassifier(ANNClassifier): {
                "module__num_classes": [num_classes],
                "module__num_features": [num_features],
                "max_epochs": [60],
                "lr": [0.05],
                "optimizer": [torch.optim.Adam],
                "verbose": [0],
                "device": ["cuda" if torch.cuda.is_available() else "cpu"],
            },
        }


class ANNRegressor(nn.Module):
    """A simple artificial neural network regressor."""

    def __init__(self, num_features):
        super(ANNRegressor, self).__init__()
        self.layer1 = nn.Linear(num_features, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
         - x (torch.Tensor): The input tensor.

        Returns:
         - torch.Tensor: The output tensor."""
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class ANNClassifier(nn.Module):
    """A simple artificial neural network classifier."""

    def __init__(self, num_features, num_classes):
        super(ANNClassifier, self).__init__()
        self.layer1 = nn.Linear(num_features, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
         - x (torch.Tensor): The input tensor.

        Returns:
         - torch.Tensor: The output tensor."""
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=1)
        return x
