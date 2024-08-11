import torch
import argparse
import models
from data import DenoiseDataset, utils, make_image_transform_crop
from tqdm import tqdm
from omegaconf import OmegaConf
import loss as losses


def train(config):
    model = models.__dict__[config.model.arch](config).to("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = losses.__dict__[config.loss.loss](config)

    optim = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    dataset = DenoiseDataset(noisy_folder=config.data.noisy_folder,
                             clean_folder=config.data.clean_folder,
                             artificial_noise=config.data.use_artificial_noise,
                             image_transform=make_image_transform_crop(),
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
        for i, (noisy, clean) in enumerate(tqdm(dataloader)):
            if torch.cuda.is_available():
                noisy = noisy.cuda(non_blocking=True)
                clean = clean.cuda(non_blocking=True)

            output = model(noisy)
            output = noisy - output

            loss = loss_fn(output, clean)

            optim.zero_grad()

            loss.backward()
            loss_avg += loss.item()

            optim.step()

        # Plotting
        utils.plot_batch(noisy, output, clean, e, num_of_images=config.data.images_to_plot, folder="plots/denoise")

        print(f"Epoch : {e}; Loss : {loss_avg / len(dataloader)}")

        if config.training.checkpointing:
            torch.save(model.state_dict(), "./model_denoise.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train GraXpertAI denoise", )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    train(config)
