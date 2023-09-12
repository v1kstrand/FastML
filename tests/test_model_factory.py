import os
import sys

import pytest
import torch
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

sys.path.append(os.getcwd())

from module.model_factory import ANNClassifier, ANNRegressor, ModelFactory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_build():
    num_features = 10
    num_classes = 3
    model_dict_clf = ModelFactory.build(True, num_features, num_classes)
    logistic, knn, svc, neural_net = model_dict_clf.keys()
    assert isinstance(logistic, LogisticRegression)
    assert isinstance(knn, KNeighborsClassifier)
    assert isinstance(svc, SVC)
    assert isinstance(neural_net, NeuralNetClassifier)
    assert model_dict_clf[neural_net]["module__num_features"] == [num_features]
    assert model_dict_clf[neural_net]["module__num_classes"] == [num_classes]

    model_dict_reg = ModelFactory.build(False, num_features, num_classes)
    linear, lasso, ridge, elastic, svr, neural_net = model_dict_reg.keys()
    assert isinstance(linear, LinearRegression)
    assert isinstance(lasso, Lasso)
    assert isinstance(ridge, Ridge)
    assert isinstance(elastic, ElasticNet)
    assert isinstance(svr, SVR)
    assert isinstance(neural_net, NeuralNetRegressor)
    assert model_dict_reg[neural_net]["device"] == [DEVICE]
    assert model_dict_reg[neural_net]["module__num_features"] == [num_features]


def test_classifier_build():
    factory = ModelFactory()
    num_features = 10
    num_classes = 3
    model_dict = factory.classifier_build(num_features, num_classes)
    logistic, knn, svc, neural_net = model_dict.keys()
    assert isinstance(logistic, LogisticRegression)
    assert isinstance(knn, KNeighborsClassifier)
    assert isinstance(svc, SVC)
    assert isinstance(neural_net, NeuralNetClassifier)
    assert "C" in model_dict[logistic]
    assert "n_neighbors" in model_dict[knn]
    assert "C" in model_dict[svc]
    assert model_dict[neural_net]["device"] == [DEVICE]
    assert model_dict[neural_net]["module__num_features"] == [num_features]
    assert model_dict[neural_net]["module__num_classes"] == [num_classes]


def test_regressor_build():
    factory = ModelFactory()
    num_features = 10
    model_dict = factory.regressor_build(num_features)
    linear, lasso, ridge, elastic, svr, neural_net = model_dict.keys()
    assert isinstance(linear, LinearRegression)
    assert isinstance(lasso, Lasso)
    assert isinstance(ridge, Ridge)
    assert isinstance(elastic, ElasticNet)
    assert isinstance(svr, SVR)
    assert isinstance(neural_net, NeuralNetRegressor)
    assert "alpha" in model_dict[lasso]
    assert "alpha" in model_dict[ridge]
    assert "alpha" in model_dict[elastic]
    assert "C" in model_dict[svr]
    assert model_dict[neural_net]["device"] == [DEVICE]
    assert model_dict[neural_net]["module__num_features"] == [num_features]


# TEST PyTorch MODELS
def test_ann_regressor_forward():
    model = ANNRegressor(10)  # 10 features, 1 output
    assert isinstance(model, torch.nn.Module)
    sample_input = torch.randn(5, 10)  # Batch size of 5, 10 features

    output = model(sample_input)
    assert output.shape == (5, 1), f"Expected (5, 1), got {output.shape}"


def test_ann_classifier_forward():
    model = ANNClassifier(10, 3)  # 10 features, 3 classes
    sample_input = torch.randn(5, 10)  # Batch size of 5, 10 features

    output = model(sample_input)
    assert output.shape == (5, 3)

    # Test if the output probabilities sums to 1  (bc of soft max)
    assert torch.allclose(output.sum(dim=1), torch.tensor([1.0] * 5))


if __name__ == "__main__":
    pytest.main([__file__])
