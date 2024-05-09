import os

import matplotlib.pyplot as plt

from .utils import median_abs_deviation, log_norm, linear_fit, lin_norm
import torch
import numpy as np
import torch.utils.data


class BkgDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, bkg_folder, image_transform, bkg_transform, epsilon=1e-6, validation=False):
        super(BkgDataset, self).__init__()

        self.image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]
        self.bkg_files = [os.path.join(bkg_folder, file) for file in os.listdir(bkg_folder)]

        self.image_transform = image_transform
        self.bkg_transform = bkg_transform

        self.epsilon = epsilon
        self.validation = validation

    def __len__(self):
        return len(self.image_files)

    # TODO : implement different median and mad for each channel
    def __getitem__(self, x):
        img_file = np.random.choice(self.image_files, 1)[0]
        bkg_file = np.random.choice(self.bkg_files, 1)[0]

        # Loading files lazily with memmap to preserve memory and time
        img = np.load(img_file, mmap_mode="c")
        bkg = np.load(bkg_file, mmap_mode="c")

        # Augmenting data
        img = self.image_transform(img)
        bkg = self.bkg_transform(bkg)

        # Normalizing background to image statistics
        median_bkg = torch.median(bkg)
        median_img = torch.median(img)
        bkg = (bkg - median_bkg) / median_abs_deviation(bkg, median_bkg) * median_abs_deviation(img, median_img)

        # Augmenting gradient intensity
        if not self.validation:
            bkg_scaling = np.power(10, np.random.uniform(-1.2, 1.2))
            # color_scaling = 0.5 * torch.randn(3, 1, 1) + 1
            bkg = bkg * bkg_scaling  # * color_scaling

        # Summing images
        img = img + bkg  # From this point forward img represents the image with the gradient added

        # Normalizing (shift)
        min_ = torch.min(img)
        img = img - min_ + self.epsilon
        bkg = torch.clip(bkg - min_ + self.epsilon, self.epsilon, torch.inf)

        # Normalizing (log transform)
        img = torch.log(img)
        bkg = torch.log(bkg)

        # Normalizing (mean and std)
        mean_img = torch.mean(img)
        std_img = torch.std(img)
        img = (img - mean_img) / std_img * 0.1
        bkg = (bkg - mean_img) / std_img * 0.1

        if self.validation:
            return img, bkg, min_, mean_img, std_img
        return img, bkg


class DenoiseDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_folder, clean_folder, image_transform, epsilon=1e-6):
        super(DenoiseDataset, self).__init__()

        self.noisy_files = [os.path.join(noisy_folder, file) for file in os.listdir(noisy_folder)]
        self.clean_folder = clean_folder

        self.image_transform = image_transform

        self.epsilon = epsilon

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, x):
        noisy_file = np.random.choice(self.noisy_files, 1)[0]
        clean_file = os.path.join(self.clean_folder, os.path.split(noisy_file)[-1])

        # Loading files lazily with memmap to preserve memory and time
        noisy = np.load(noisy_file, mmap_mode="c")
        clean = np.load(clean_file, mmap_mode="c")

        # Augmenting data
        state_torch = torch.get_rng_state()
        state_numpy = np.random.get_state()
        noisy = self.image_transform(noisy)
        torch.set_rng_state(state_torch)
        np.random.set_state(state_numpy)
        clean = self.image_transform(clean)

        clean = linear_fit(noisy, clean)

        return noisy, clean
