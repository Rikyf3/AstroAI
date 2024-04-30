import torch
import matplotlib.pyplot as plt


def median_abs_deviation(x, median_x):
    x = torch.median(torch.abs(x - median_x))
    return x


def stretch(x, normalize=False):
    if normalize:
        med = torch.median(x)
        x = (x - med) / median_abs_deviation(x, med) * 0.04

    return torch.clip(x.permute(1, 2, 0) * 3 + 0.2, 0.0, 1.0)


@torch.no_grad()
def plot_batch(image, output, bkg, epoch, num_of_images=8):
    fig, axs = plt.subplots(num_of_images, 3, figsize=(16.5, 30.5))

    for i in range(num_of_images):
        axs[i, 0].imshow(stretch(image[i].cpu(), normalize=True), interpolation="lanczos")
        axs[i, 1].imshow(stretch(output[i].cpu(), normalize=True), interpolation="bilinear")
        axs[i, 2].imshow(stretch(bkg[i].cpu(), normalize=True), interpolation="bilinear")

    fig.savefig(f"plot_{epoch}", bbox_inches='tight')
    plt.close(fig)


def log_norm(x, min_=None, mean_=None, std_=None, epsilon=1e-5, clip=False):
    min_ = torch.min(x) if min_ is None else min_

    x = x - min_ + epsilon if not clip else torch.clip(x - min_ + epsilon, epsilon, torch.inf)
    x = torch.log(x)

    mean_ = torch.mean(x) if mean_ is None else mean_
    std_ = torch.std(x) if std_ is None else std_

    x = (x - mean_) / std_ * 0.1
    x = torch.nan_to_num(x, 0)

    return x, min_, mean_, std_


def log_denorm(x, min_, mean_, std_, epsilon=1e-5):
    x = x * std_ / 0.1 + mean_
    x = torch.exp(x)
    x = x + min_ - epsilon

    return x
