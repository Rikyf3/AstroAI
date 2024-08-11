import os.path

import torch
import argparse
import models
import loss as losses
from data import BkgDataset, make_image_transform_crop_resize, make_bkg_transform, make_transform_val
from data.utils import lin_denorm, plot_batch
import torch.utils.data
from tqdm import tqdm
from omegaconf import OmegaConf


def train(config):
    model = models.__dict__[config.model.arch](config).to("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = losses.__dict__[config.loss.loss](config)

    optim = torch.optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=1e-4)

    dataset = BkgDataset(image_folder=config.data.image_folder,
                         bkg_folder=config.data.bkg_folder,
                         image_transform=make_image_transform_crop_resize(),
                         bkg_transform=make_bkg_transform(),
                         )

    sampler = torch.utils.data.RandomSampler(dataset,
                                             replacement=True,
                                             num_samples=config.training.iters_per_epoch * config.training.batch_size)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config.training.batch_size,
                                             sampler=sampler,
                                             num_workers=config.data.num_workers,
                                             )

    for e in range(args.epochs):
        loss_avg = 0.0

        # Training cycle
        model.train()
        for i, (image, bkg) in enumerate(tqdm(dataloader)):
            if torch.cuda.is_available():
                image = image.cuda(non_blocking=True)
                bkg = bkg.cuda(non_blocking=True)

            output = model(image)
            loss = loss_fn(output, bkg)

            optim.zero_grad()

            loss.backward()
            loss_avg += loss.item()

            optim.step()

        # Plotting
        plot_batch(image, output, bkg, e, num_of_images=config.data.images_to_plot, folder="plots/bge")

        print(f"Epoch : {e}; Loss : {loss_avg / len(dataloader)}")

        if config.training.checkpointing:
            torch.save(model.state_dict(), "./model_bge.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train GraXpertAI background extractor")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    train(config)
