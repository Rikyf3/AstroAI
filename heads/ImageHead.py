import torch
import torch.nn as nn
import math
from torch.nn.init import trunc_normal_


# TODO : add confidence score with [CLS]
class ImageHead(torch.nn.Module):
    def __init__(self, token_size=768, output_size=588):
        super().__init__()

        self.token_size = token_size
        self.output_size = output_size
        self.output_hw = int(math.sqrt(output_size // 3))

        self.fc1 = nn.Linear(self.token_size, 4*self.token_size)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(4*self.token_size, self.output_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, C = x.shape

        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)

        x = x.reshape(B, N, 3, self.output_hw, self.output_hw)

        x = x.view(B, 16, 16, 3, 14, 14)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 3, 224, 224)

        return x
