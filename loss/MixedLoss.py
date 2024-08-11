import torch
import torchmetrics
import numpy as np


class MixedLoss(torch.nn.Module):
    def __init__(self, alpha_start, alpha_min, max_iterations):
        super(MixedLoss, self).__init__()

        self.loss_1 = torch.nn.MSELoss()
        self.loss_2 = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to("cuda" if torch.cuda.is_available() else "cpu")

        self.current_iter = 0
        self.alpha = alpha_start
        self.alpha_start = alpha_start
        self.alpha_min = alpha_min
        self.max_iterations = max_iterations

    def step(self):
        self.current_iter += 1

        if self.current_iter <= self.max_iterations:
            self.alpha = self.alpha_min + 0.5*(self.alpha_start - self.alpha_min)*(1 + np.cos(self.current_iter/self.max_iterations*np.pi))

    def get_alpha(self):
        return self.alpha

    def forward(self, x, y):
        return self.alpha * (1 - self.loss_2(x, y)) + (1 - self.alpha) * self.loss_1(x, y)
