from acse_softmax import data_loader, main
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import sys

sys.path.append(".")


def test_trainimagesize():
    """
    Tests the read in train image size
    """
    train = data_loader.load_training_data("./data/train")
    shape = (224, 224)
    for t in train:
        assert (shape == t[0].shape)


def test_testimagesize():
    """
    Tests the read in test image size
    """
    test, _ = data_loader.load_test_data("./data/test")
    shape = (224, 224)
    for t in test:
        assert (shape == t[0].shape)


def test_datasize():
    """
    Tests the read in datasize
    """
    xray_train, xray_validate = main.make_dataset(True, './data/train')
    for tens in xray_train:
        assert (224 == tens[0].shape[1])
        assert (224 == tens[0].shape[2])
    for tens in xray_validate:
        assert (224 == tens[0].shape[1])
        assert (224 == tens[0].shape[2])


def test_normalize():
    """
    Tests the normalization function
    """
    input_image = np.array([[200., 150.], [100., 50.]])
    print(np.std(input_image))
    expected_image = np.array([[0.78922445, 0.], [-0.78922445, -1.5784491]])
    result = main.apply_normalisation(input_image, mean=150 / 255., std=55.9 / 225)
    result = result.numpy()
    assert np.allclose(result, expected_image)


def test_shape():
    """
    Tests the output shape of the model with specified inpput shape
    """
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    X = torch.randn(1, 1, 224, 224)
    output = model(X)
    assert (torch.Size([1, 1000]) == output.size())


def test_grad():
    """
    Tests the gradient of the parameters
    The gradient of the parameters should not be none and should not equal to 0
    """
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    X = torch.randn(4, 1, 224, 224)
    y = torch.ones(4).long()
    y[1] = 0
    y[2] = 2
    a2 = model(X.reshape(-1, 1, 224, 224))
    loss = criterion(a2, y)
    loss.backward()
    optimizer.step()
    for filters in model.parameters():
        if filters.requires_grad:
            assert (filters.grad is not None)
            assert (0 != torch.sum(filters.grad ** 2))


def test_loss():
    """
    Tests the loss of the model which cannot be zero
    """
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    X = torch.randn(4, 1, 224, 224)
    y = torch.ones(4).long()
    y[1] = 0
    y[2] = 2
    a2 = model(X.reshape(-1, 1, 224, 224))
    loss = criterion(a2, y)
    loss.backward()
    optimizer.step()
    assert (loss != 0)


def test_para():
    """
    Tests the parameters of the model changes during training
    """
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    before = model.parameters()

    model.train()
    optimizer.zero_grad()
    X = torch.randn(4, 1, 224, 224)
    y = torch.ones(4).long()
    y[1] = 0
    y[2] = 2
    a2 = model(X.reshape(-1, 1, 224, 224))
    loss = criterion(a2, y)
    loss.backward()
    optimizer.step()

    after = model.parameters()
    for b, a in zip(before, after):
        assert (before != after)
