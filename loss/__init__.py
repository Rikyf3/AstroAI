from .MixedLoss import MixedLoss
import torch


def mixedloss(config):
    return MixedLoss(alpha_start=config.loss.alpha_start,
                     alpha_min=config.loss.alpha_min,
                     max_iterations=config.loss.max_iterations,
                     )


def l1(config):
    return torch.nn.L1Loss()


def l2(config):
    return torch.nn.MSELoss()