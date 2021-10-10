from acse_softmax import data_loader

import argparse
import random
import numpy as np
import wandb
import time
import torch
from torchvision import models, transforms
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn as nn

__all__ = ['set_hyperparameters', 'set_up', 'set_seed', 'set_device',
           'apply_normalisation', 'make_dataset', 'make_dataloader',
           'train', 'validate', 'evaluate', 'train_model']

from acse_softmax.custom_dataset import CustomImageTensorDataset


def set_hyperparameters(args):
    # Set training hyperparameters
    hyperparameters = dict(
        # random seed(default:42)
        seed=args.s,
        # number of epochs to train(default:10)
        epochs=args.e,
        # input batch size for training (default:64)
        batch_size=args.bs,
        # input batch size for testing(default:1000)
        test_batch_size=args.ts,
        # learning rate(default:0.01)
        learning_rate=args.lr,
        model_name="ResNet18",
        criterion_name="CrossEntropyLoss",
        optimizer_name="SGD",
        valid_split=0.2,
        train_transform=None,
        test_transform=None,
        # learning rate(default:0.01)
        momentum=0.5,
        weight_decay=1e-4,
        # disables CUDA training
        no_cuda=False,
        # how many batches to wait before logging training status
        log_interval=10,
    )
    return hyperparameters


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def set_device():
    global device
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = 'cuda'
        print("Cuda installed! Running on GPU %s!" % torch.cuda.get_device_name())

    else:
        print("No GPU available! Running on CPU")
    return True


def apply_normalisation(X, mean, std):
    """
    Normalise the set X
    input: X, an array of numbers that is ready to be normalised
    output: normalised X
    """
    X = torch.Tensor(X)

    X /= 255.
    X -= mean
    X /= std
    return X


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(224, padding=20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.509,), (0.253,))
])

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.509,), (0.253,))
])


def make_dataset(train, file_path, download=True):
    if train:
        train_set = data_loader.load_training_data(file_path)
        train_data = []
        train_targets = []
        for feature, label in train_set:
            train_data.append(feature)
            train_targets.append(label)

        train_data = np.array(train_data)

        shuffler = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=42).split(train_data,
                                                                                                            train_targets)
        indices = [(train_idx, validation_idx) for train_idx, validation_idx in shuffler][0]

        mean = train_data.mean() / 255
        std = train_data.std() / 255

        X_train, y_train = apply_normalisation(train_data[indices[0]], mean, std), torch.Tensor(train_targets)[
            indices[0]]
        X_val, y_val = apply_normalisation(train_data[indices[1]], mean, std), torch.Tensor(train_targets)[indices[1]]

        xray_train = CustomImageTensorDataset(X_train, y_train.long(), transform=train_transform)
        xray_validate = CustomImageTensorDataset(X_val, y_val.long(), transform=val_test_transform)

        return xray_train, xray_validate

    else:
        test_set, test_names = data_loader.load_test_data(file_path)
        test_data = []
        test_targets = []
        for feature, label in test_set:
            test_data.append(feature)
            test_targets.append(label)

        test_data = np.array(test_data)
        mean = test_data.mean() / 255
        std = test_data.std() / 255
        X_test, y_test = apply_normalisation(test_data, mean, std), torch.Tensor(test_targets)
        xray_test = CustomImageTensorDataset(X_test, y_test.long(), transform=val_test_transform)
        return xray_test


def make_dataloader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=8):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=pin_memory, num_workers=num_workers)
    return loader


def set_up(config):
    # Get data and make datasets
    trainds, validds = make_dataset(train=True, file_path='./xray-data/xray-data/train', download=True)
    testds = make_dataset(train=False, file_path='./xray-data/xray-data/test', download=True)

    # Get data loaders
    train_loader = make_dataloader(trainds, config.batch_size)
    valid_loader = make_dataloader(validds, config.test_batch_size)
    test_loader = make_dataloader(testds, config.test_batch_size)

    # Make model
    try:
        model = models.resnet18()
        # Since our image is in grayscale format, we need to change the channel of the layer to 1.
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    except:
        raise NotImplementedError("Model of name %s has not been found in this file" % config.model_name)
    config.model = model

    # Make optimizer
    try:
        optimizer = getattr(torch.optim, config.optimizer_name)
        optimizer = optimizer(model.parameters(), lr=config.learning_rate)

    except:
        raise NotImplementedError("Optimizer of name %s has not been found in torch.optim" % config.optimizer_name)
    try:
        for g in optimizer.param_groups:
            g['momentum'] = config.momentum
    except:
        config.momentum = 0
        pass
    config.optimizer = optimizer

    # Make loss
    try:
        criterion = getattr(torch.nn, config.criterion_name)
        criterion = criterion()
    except:
        raise NotImplementedError("Criterion of name %s has not been found in torch.nn" % config.criterion_name)
    config.criterion = criterion

    return model, criterion, optimizer, train_loader, valid_loader, test_loader


def train(config, model, optimizer, criterion, data_loader):
    """
    Implement train function.
    input:
    config, config paramters of wandb.
    model, model to be trained.
    optimiser, optimisation method.
    criterion, a function that calculates loss.
    data_loader, torch.Tensor, data to be input.
    output:
    train_loss, float, loss of training dataset.
    train_accuracy, float, loss of training dataset.
    """
    # switch model to training mode
    model.train()
    train_loss, train_accuracy = 0, 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()
        a2 = model(X.reshape(-1, 1, 224, 224))
        loss = criterion(a2, y)
        loss.backward()
        train_loss += loss * X.size(0)
        y_pred = F.log_softmax(a2, dim=1).max(1)[1]
        train_accuracy += accuracy_score(y.cpu().numpy(), y_pred.detach().cpu().numpy()) * X.size(0)
        optimizer.step()

    return train_loss / len(data_loader.dataset), train_accuracy / len(data_loader.dataset)


def validate(model, criterion, data_loader):
    """
    Implement validate function.
    input:
    model, model to be trained.
    optimiser, optimisation method.
    criterion, a function that calculates loss.
    data_loader, torch.Tensor, data to be input.
    output:
    validation_loss, float, loss of validation dataset.
    validation_accuracy, float, accuracy of validation dataset.
    """
    model.eval()
    validation_loss, validation_accuracy = 0., 0.
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.reshape(-1, 1, 224, 224))
            loss = criterion(a2, y)
            validation_loss += loss * X.size(0)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            validation_accuracy += accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy()) * X.size(0)

    return validation_loss / len(data_loader.dataset), validation_accuracy / len(data_loader.dataset)


def evaluate(model, data_loader):
    """
    Implement evaluate function.
    input:
    model, model have been trained.
    data_loader, torch.Tensor, data to be input.
    output:
    y_preds, numpy.array, predicted labels.
    ys, numpy.array, actual labels.
    """
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.reshape(-1, 1, 224, 224))
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    return np.concatenate(y_preds, 0), np.concatenate(ys, 0)


def train_model(hyperparameters):
    """
    Training model with this function.
    intput:
    plot:, boolean, to tell whether live update is plotted or not.
    output:
    model, final model trained.
    """

    # todo delete hard code
    wandb.login(key='8ebeb3b9dc218ff686b10d97c906d8a84b6d0a1a')

    # Initialize a new run and set hyperparameters
    # tell wandb to get started
    with wandb.init(config=hyperparameters, project='ResNet18', entity='x-ray-classification-softmax'):
        # Set seed and devices
        set_seed(hyperparameters['seed'])
        set_device()

        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Set up run with hypeparameters
        model, criterion, optimizer, train_loader, valid_loader, test_loader = set_up(config)

        # Let wandb watch the model and the criterion
        wandb.watch(model, criterion)

        for epoch in range(config.epochs):
            train_loss, train_accuracy = train(model, optimizer, criterion, train_loader)
            validation_loss, validation_accuracy = validate(model, criterion, valid_loader)
            # wandb.log is used to record some logs (accuracy, loss and epoch),
            # so that you can check the performance of the network at any time
            log = {"epoch": epoch + 1, "train_loss": train_loss.item(), "train_accuracy": train_accuracy.item(),
                   "valid_loss": validation_loss.item(), "valid_accuracy": validation_accuracy.item()}
            print(log)
            wandb.log(log)

    model_save_name = 'ResNet18_X-ray_classifier_{}.pt'.format(int(round(time.time() * 1000)))
    torch.save(model.state_dict(), model_save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Covid_CT_Classifier_Softmax')
    parser.add_argument('-version', action='version', version='%(prog)s 1.0')
    parser.add_argument('-s', default=42, help='seed')
    parser.add_argument('-lr', default=1e-2, help='learning rate')
    parser.add_argument('-m', default=0.5, help='momentum')
    parser.add_argument('-bs', default=32, help='batch size')
    parser.add_argument('-ts', default=32, help='test batch size')
    parser.add_argument('-e', default=10, help='epoch')
    args = parser.parse_args()

    hyperparameters = set_hyperparameters(args)
    train_model(hyperparameters)
