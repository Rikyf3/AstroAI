import os
import logging
import hashlib
import argparse
import numpy as np
import skimage.transform
from tqdm import tqdm
from astropy.io import fits


def main(args):
    assert os.path.isdir(args.img_folder) and os.path.isdir(args.bkg_folder), f"{args.img_folder} or {args.bkg_folder} are not folders"

    output_img_folder = os.path.join(args.output_folder, "img")
    output_bkg_folder = os.path.join(args.output_folder, "bkg")
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_bkg_folder, exist_ok=True)

    if max(len(os.listdir(output_img_folder)), len(os.listdir(output_bkg_folder))) > 0:
        logging.warning(f"{output_img_folder} and/or {output_bkg_folder} are not empty")

    img_files = sorted([file for file in os.listdir(args.img_folder) if file.endswith(".fits")])
    bkg_files = sorted([file for file in os.listdir(args.bkg_folder) if file.endswith(".fits")])

    assert len(img_files) == len(bkg_files), "Folders do not contain the same number of files"
    num_files = len(img_files)

    assert sorted(img_files) == sorted(bkg_files), "Files mismatch in folders"

    m = hashlib.md5()
    progess = tqdm(total=num_files)
    for img_file, bkg_file in zip(img_files, bkg_files):
        img_path = os.path.join(args.img_folder, img_file)
        bkg_path = os.path.join(args.bkg_folder, bkg_file)

        # Load fits
        img = fits.getdata(img_path, ext=0)
        bkg = fits.getdata(bkg_path, ext=0)

        if img.shape != bkg.shape:
            logging.error(f"{img_file} and {bkg_file} are not the same photo! Shapes do not match.")
            continue

        # Channel first conversion
        img = np.moveaxis(img, 0, 2)
        bkg = np.moveaxis(bkg, 0, 2)

        # Subtract
        img = img - bkg

        # Resize
        img = skimage.transform.resize(img, output_shape=(args.output_size, args.output_size), order=2, anti_aliasing=True)
        bkg = skimage.transform.resize(bkg, output_shape=(args.output_size, args.output_size), order=2, anti_aliasing=True)

        # Change channel position
        img = np.moveaxis(img, -1, 0)
        bkg = np.moveaxis(bkg, -1, 0)

        img = img.astype(np.float32)
        bkg = bkg.astype(np.float32)

        # Save images in .npy (use np.memmap to open it)
        m.update(bytes(img_file, encoding='utf8'))
        filename = f"{m.hexdigest()}.npy"
        np.save(os.path.join(args.output_folder, "img", filename), img)
        np.save(os.path.join(args.output_folder, "nkg", filename), bkg)

        progess.update(1)

    progess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="prepare_fits",
        description="Prepares .fits for training by repacking them in .npy files: an easier format to handle."
    )

    parser.add_argument("--img_folder", required=True, type=str)
    parser.add_argument("--bkg_folder", required=True, type=str)
    parser.add_argument("--output_folder", required=True, type=str)
    parser.add_argument("--output_size", required=False, type=int, default=1024)
    args = parser.parse_args()

    main(args)
