import os
from .utils import median_abs_deviation, log_norm, linear_fit, lin_norm
import torch
import numpy as np
import torch.utils.data
from scipy import stats


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
            bkg = bkg * bkg_scaling

        # Summing images
        img = img + bkg  # From this point forward img represents the image with the gradient added

        img, med_, mad_ = lin_norm(img)
        bkg, _, _ = lin_norm(bkg, med_=med_, mad_=mad_)

        if self.validation:
            return img, bkg, med_, mad_
        return img, bkg


class DenoiseDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_folder, clean_folder, image_transform, artificial_noise=True):
        super(DenoiseDataset, self).__init__()

        self.clean_files = [os.path.join(clean_folder, file) for file in os.listdir(clean_folder)]
        self.noisy_folder = noisy_folder

        self.image_transform = image_transform

        self.artificial_noise = artificial_noise

    def __len__(self):
        return len(self.clean_files)

    def _getitem_natural(self, noisy_file, clean_file):
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

    def _getitem_artificial(self, clean_file):
        # Loading files lazily with memmap to preserve memory and time
        clean = np.load(clean_file, mmap_mode="c")

        # Augmenting data
        _mad = stats.median_abs_deviation(clean, axis=[-1, -2])[:, None, None]
        clean = self.image_transform(clean)

        # Adding noise
        noise = 1.4826 * torch.from_numpy(_mad) * torch.randn_like(clean)
        noisy = clean + noise

        return noisy, clean

    def __getitem__(self, x):
        clean_file = np.random.choice(self.clean_files, 1)[0]
        noisy_file = os.path.join(self.noisy_folder, os.path.split(clean_file)[-1])

        if self.artificial_noise:
            noisy, clean = self._getitem_artificial(clean_file)
        else:
            noisy, clean = self._getitem_natural(noisy_file, clean_file)

        return noisy, clean
