import os

import torch
import scipy
import matplotlib.pyplot as plt


def median_abs_deviation(x, median_x):
    x = torch.abs(x - median_x)
    x = torch.median(x.reshape(x.shape[0], -1), dim=1).values[:, None, None]
    return x


def stretch(x, normalize=False):
    if normalize:
        med = torch.median(x)
        x = (x - med) / median_abs_deviation(x, med) * 0.04

    return torch.clip(x.permute(1, 2, 0) * 3 + 0.3, 0.0, 1.0)


@torch.no_grad()
def plot_batch(image, output, bkg, epoch, num_of_images=8, folder="./"):
    fig, axs = plt.subplots(num_of_images, 3, figsize=(16.5, 30.5))

    if num_of_images == 1:
        axs[0].imshow(stretch(image[0].cpu(), normalize=False), interpolation="lanczos", cmap="grey", vmin=0, vmax=1)
        axs[1].imshow(stretch(output[0].cpu(), normalize=False), interpolation="bilinear", cmap="grey", vmin=0, vmax=1)
        axs[2].imshow(stretch(bkg[0].cpu(), normalize=False), interpolation="bilinear", cmap="grey", vmin=0, vmax=1)
    else:
        for i in range(num_of_images):
            axs[i, 0].imshow(stretch(image[i].cpu(), normalize=False), interpolation="lanczos", cmap="grey", vmin=0, vmax=1)
            axs[i, 1].imshow(stretch(output[i].cpu(), normalize=False), interpolation="bilinear", cmap="grey", vmin=0, vmax=1)
            axs[i, 2].imshow(stretch(bkg[i].cpu(), normalize=False), interpolation="bilinear", cmap="grey", vmin=0, vmax=1)

    fig.savefig(os.path.join(folder, f"plot_{epoch}.png"), bbox_inches='tight')
    plt.close(fig)


def log_norm(x, min_=None, mean_=None, std_=None, epsilon=1e-5, clip=False):
    min_ = torch.min(x.reshape(x.shape[0], -1), dim=1).values[:, None, None] if min_ is None else min_

    x = x - min_ + epsilon
    if clip:
        x = torch.clip(x, epsilon, torch.inf)

    x = torch.log(x)

    mean_ = torch.mean(x.reshape(x.shape[0], -1), dim=1)[:, None, None] if mean_ is None else mean_
    std_ = torch.std(x.reshape(x.shape[0], -1), dim=1)[:, None, None] if std_ is None else std_

    x = (x - mean_) / std_ * 0.1
    x = torch.nan_to_num(x, 0)

    return x, min_, mean_, std_


def log_denorm(x, min_, mean_, std_, epsilon=1e-5):
    x = x * std_ / 0.1 + mean_
    x = torch.exp(x)
    x = x + min_ - epsilon

    return x


def lin_norm(x, med_=None, mad_=None):
    med_ = torch.median(x.reshape(x.shape[0], -1), dim=1).values[:, None, None] if med_ is None else med_
    mad_ = median_abs_deviation(x, med_) if mad_ is None else mad_

    x = (x - med_) / mad_ * 0.04
    x = torch.clip(x, -10, 10)

    return x, med_, mad_


def lin_denorm(x, med_, mad_):
    return (x + med_) * mad_ / 0.04


def linear_fit(x, y, clipping=0.95):
    x, y = x.numpy(), y.numpy()

    def f(x_in, slope, intercept):
        return x_in * slope + intercept

    for c in range(x.shape[0]):
        indx_clipped = x[c, :, :].flatten() < clipping
        popt, _ = scipy.optimize.curve_fit(f, y[c, :, :].flatten()[indx_clipped], x[c, :, :].flatten()[indx_clipped], p0=(1, 0), method='lm')
        y[c, :, :] = y[c, :, :] * popt[0] + popt[1]

    return y
