import os
import logging
import hashlib
import argparse
import numpy as np
import torch
from tqdm import tqdm
from astropy.io import fits
from utils import lin_norm


def main(args):
    if args.artifical_noise:
        prepare_artifical_noise(args)
    else:
        prepare_real_noise(args)
        

def prepare_real_noise(args):
    assert os.path.isdir(args.noisy_folder) and os.path.isdir(args.clean_folder), f"{args.noisy_folder} or {args.clean_folder} are not folders"

    output_noisy_folder = os.path.join(args.output_folder, "noisy")
    output_clean_folder = os.path.join(args.output_folder, "clean")
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(output_noisy_folder, exist_ok=True)
    os.makedirs(output_clean_folder, exist_ok=True)

    if max(len(os.listdir(output_noisy_folder)), len(os.listdir(output_clean_folder))) > 0:
        logging.warning(f"{output_noisy_folder} and/or {output_clean_folder} are not empty")

    noisy_files = sorted([file for file in os.listdir(args.noisy_folder) if file.endswith(".fits")])
    clean_files = sorted([file for file in os.listdir(args.clean_folder) if file.endswith(".fits")])

    assert len(noisy_files) == len(clean_files), "Folders do not contain the same number of files"
    num_files = len(noisy_files)

    assert sorted(noisy_files) == sorted(noisy_files), "Files mismatch in folders"

    m = hashlib.md5()
    progess = tqdm(total=num_files)
    for noisy_file, clean_file in zip(noisy_files, clean_files):
        noisy_path = os.path.join(args.noisy_folder, noisy_file)
        clean_path = os.path.join(args.clean_folder, clean_file)

        # Load fits
        noisy = fits.getdata(noisy_path, ext=0)
        clean = fits.getdata(clean_path, ext=0)

        if noisy.shape != clean.shape:
            logging.error(f"{noisy_file} and {clean_file} are not the same photo! Shapes do not match.")
            continue

        # Cast to float32
        noisy = noisy.astype(np.float32)
        clean = clean.astype(np.float32)

        # Normalization
        noisy, med_, mad_ = lin_norm(torch.from_numpy(noisy))
        clean, _, _ = lin_norm(torch.from_numpy(clean), med_, mad_)
        noisy = noisy.numpy()
        clean = clean.numpy()

        # Save images in .npy (use np.memmap to open it)
        m.update(bytes(noisy_file, encoding='utf8'))
        filename = f"{m.hexdigest()}.npy"
        np.save(os.path.join(args.output_folder, "noisy", filename), noisy)
        np.save(os.path.join(args.output_folder, "clean", filename), clean)

        progess.update(1)

    progess.close()
    
def prepare_artifical_noise(args):
    assert os.path.isdir(args.clean_folder), f"{args.clean_folder} is not a folder"

    output_clean_folder = os.path.join(args.output_folder, "clean")
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(output_clean_folder, exist_ok=True)

    if len(os.listdir(output_clean_folder)) > 0:
        logging.warning(f"{output_clean_folder} is not empty")

    clean_files = sorted([file for file in os.listdir(args.clean_folder) if file.endswith(".fits")])
    num_files = len(clean_files)

    m = hashlib.md5()
    progess = tqdm(total=num_files)
    for clean_file in clean_files:
        clean_path = os.path.join(args.clean_folder, clean_file)
        clean = fits.getdata(clean_path, ext=0)
        H, W, num_channels = clean.shape
        
        mad = []
        fits_header = fits.open(clean_path)[0].header
        for c in range(num_channels):
            mad_channel = fits_header["mad" + str(c+1)]
            print(mad_channel)
            mad.append(mad_channel)


        clean = clean.astype(np.float32)
        noisy = np.copy(clean)
        for c in range(num_channels):
            noisy[:,:,c] = clean[:,:,c] + mad[c] * args.noise_factor * np.random.normal(loc=0.0, scale=0.6745, size=(H,W))
        
        noisy, med_, mad_ =  lin_norm(torch.from_numpy(noisy), None, None)
        clean, _, _ = lin_norm(torch.from_numpy(clean), med_, mad_)
        clean = clean.numpy()

        # Save images in .npy (use np.memmap to open it)
        m.update(bytes(clean_file, encoding='utf8'))
        filename = f"{m.hexdigest()}.npy"
        np.save(os.path.join(args.output_folder, "clean", filename), clean)

        progess.update(1)

    progess.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="prepare_fits",
        description="Prepares .fits for training by repacking them in .npy files: an easier format to handle."
    )

    parser.add_argument("--noisy_folder", required=True, type=str)
    parser.add_argument("--clean_folder", required=True, type=str)
    parser.add_argument("--output_folder", required=True, type=str)
    parser.add_argument("--artifical_noise", type=bool, default=True)
    parser.add_argument("--noise_factor", type=float, required=False, default = 3.0)
    args = parser.parse_args()

    main(args)
