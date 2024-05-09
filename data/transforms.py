import torch
import numpy as np
from torchvision.transforms import v2
from .utils import median_abs_deviation


class MemEffRandomCrop(torch.nn.Module):
    def __init__(self, scale=(0.4, 1.0)):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        input_size = x.shape

        size = np.random.randint(self.scale[0] * input_size[1], self.scale[1] * input_size[1]) if type(self.scale) == tuple else self.scale

        h = np.random.randint(0, input_size[1] - size)
        w = np.random.randint(0, input_size[2] - size)

        x = torch.from_numpy(x[:, h:h+size, w:w+size])

        return x


class RandomChannelSwap(torch.nn.Module):
    def __init__(self):
        super(RandomChannelSwap, self).__init__()

    def forward(self, x):
        swap = torch.randperm(x.shape[0])

        return x[swap, :, :]


class RandomnNintyDegreeRotation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.rot90(x, np.random.randint(0, 4), dims=[-1, -2])
        return x


class FourierRandomGradient(torch.nn.Module):
    def __init__(self, p):
        super(FourierRandomGradient, self).__init__()

        self.p = p
        self.dim = 2

    def forward(self, x):
        if np.random.random() < self.p:
            syn_gradient = torch.zeros_like(x[0], dtype=torch.complex64)

            ampl = 3 * torch.randn(self.dim, self.dim)
            phase = 2 * torch.pi * torch.rand(self.dim, self.dim)
            syn_gradient[:self.dim, :self.dim] = ampl * torch.exp(1j*phase)

            syn_gradient = torch.fft.ifft2(syn_gradient).real

            syn_gradient = syn_gradient.expand(3, -1, -1)

            color_scaling = torch.rand(3, 1, 1)
            syn_gradient = syn_gradient * color_scaling

            syn_gradient_med = torch.median(syn_gradient)
            syn_gradient_mad = median_abs_deviation(syn_gradient, syn_gradient_med)
            x_med = torch.median(x)
            x_mad = median_abs_deviation(x, x_med)

            syn_gradient = (syn_gradient - syn_gradient_med) / syn_gradient_mad * 0.5
            x = (x - x_med) / x_mad

            return (syn_gradient + x) * x_mad + x_med
        else:
            return x


# Bkg transforms
def make_image_transform_crop_resize(image_output_size):
    return v2.Compose([
        MemEffRandomCrop(),
        v2.Resize(image_output_size),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        RandomnNintyDegreeRotation(),  # TODO : make it for all angles and not only 90*k
        RandomChannelSwap(),
    ])


def make_bkg_transform(image_output_size):
    return v2.Compose([
        MemEffRandomCrop(),
        v2.Resize(image_output_size),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        RandomnNintyDegreeRotation(),  # TODO : make it for all angles and not only 90*k
        v2.ColorJitter(hue=0.5, saturation=0.2),
        RandomChannelSwap(),
    ])


def make_transform_val(image_output_size):
    return v2.Compose([
        torch.from_numpy,
        v2.Resize(image_output_size),
    ])


def make_image_transform_crop():
    return v2.Compose([
        MemEffRandomCrop(scale=256),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        RandomnNintyDegreeRotation(),
    ])
