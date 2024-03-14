import torch


def calculate_accuracy(y_prob, y):
    y = y.float()
    y_hat = torch.ge(y_prob, 0.5).float()
    acc = y_hat.eq(y).cpu().float().mean().data
    return acc


