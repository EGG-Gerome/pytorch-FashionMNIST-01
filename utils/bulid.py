# utils/build.py
import torch.nn as nn
import torch

def build_optimizer(model, optimizer_type, lr):
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), lr = lr)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def build_loss(loss_type):
    if loss_type == "CrossEntropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "MSE":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")
