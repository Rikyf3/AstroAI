import torch
import argparse
from models import UNet
from data import DenoiseDataset, utils, make_image_transform_crop
from tqdm import tqdm
import torchmetrics


def train(args):
    model = UNet([64, 128, 256, 512, 512, 512, 512, 512])  # .cuda()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    loss_fn = torch.nn.L1Loss()
    ssim_fn = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr_fn = torchmetrics.image.PeakSignalNoiseRatio()

    dataset = DenoiseDataset(noisy_folder="./dataset/train/noisy",
                             clean_folder="./dataset/train/clean",
                             artificial_noise=args.artificial_noise,
                             image_transform=make_image_transform_crop(),
                             )
    dataset_val = DenoiseDataset(noisy_folder="./dataset/val/noisy",
                                 clean_folder="./dataset/val/clean",
                                 artificial_noise=False,
                                 image_transform=make_image_transform_crop(),
                                 )

    sampler = torch.utils.data.RandomSampler(dataset,
                                             replacement=True,
                                             num_samples=args.iters_per_epoch * args.batch_size)
    sampler_val = torch.utils.data.RandomSampler(dataset_val,
                                                 replacement=True,
                                                 num_samples=args.val_iters * 4)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             sampler=sampler,
                                             # num_workers=10,
                                             )
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=4,
                                                 sampler=sampler_val,
                                                 )

    for e in range(args.epochs):
        loss_avg = 0.0
        loss_avg_val = 0.0
        ssim_avg = 0.0
        psnr_avg = 0.0

        # Training cycle
        model.train()
        for i, (noisy, clean) in enumerate(tqdm(dataloader)):
            # noisy = noisy.cuda(non_blocking=True)
            # clean = clean.cuda(non_blocking=True)

            output = model(noisy)
            output = noisy - output

            loss = loss_fn(output, clean)

            optim.zero_grad()

            loss.backward()
            loss_avg += loss.item()

            optim.step()

        # Validation cycle
        model.eval()
        for noisy, clean in tqdm(dataloader_val):
            with torch.no_grad():
                output = model(noisy)
                output = noisy - output

                loss = loss_fn(output, clean)
                ssim = ssim_fn(output, clean)
                psnr = psnr_fn(output, clean)

                loss_avg_val += loss.item()
                ssim_avg += ssim.item()
                psnr_avg += psnr.item()

        # Plotting
        utils.plot_batch(noisy, output, clean, e, num_of_images=1, folder="plots/denoise")

        print(f"Epoch : {e}; Loss : {loss_avg / len(dataloader)}; Val Loss : {loss_avg_val / len(dataloader_val)}; SSIM : {ssim_avg / len(dataloader_val)}; PNSR : {psnr / len(dataloader_val)}")

        if args.save_model:
            torch.save(model.state_dict(), "./model_denoise.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train GraXpertAI denoise",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--iters_per_epoch", type=int, default=50)
    parser.add_argument("--val_iters", type=int, default=50)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--artificial_noise", action="store_true")

    args = parser.parse_args()

    train(args)
