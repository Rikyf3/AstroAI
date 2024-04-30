import os.path

import torch
import argparse
from models import VisionTransformer
from heads import ImageHead
from data import BkgDataset, make_image_transform_crop_resize, make_bkg_transform, make_transform_val, utils
import torch.utils.data
from tqdm import tqdm


def train(args):
    model = VisionTransformer(backbone_size="base", head_layer=ImageHead)  # .cuda()

    optim = torch.optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.iters_per_epoch * args.epochs,
                                                           eta_min=args.final_lr)

    loss_fn = torch.nn.L1Loss()
    loss_fn_val = torch.nn.L1Loss()

    dataset = BkgDataset(image_folder=os.path.join(args.dataset_folder, "img"),
                         bkg_folder=os.path.join(args.dataset_folder, "bkg"),
                         image_transform=make_image_transform_crop_resize(image_output_size=224),
                         bkg_transform=make_bkg_transform(image_output_size=224),
                         )

    val_dataset = BkgDataset(image_folder=os.path.join(args.validation_folder, "img"),
                             bkg_folder=os.path.join(args.validation_folder, "bkg"),
                             image_transform=make_transform_val(image_output_size=224),
                             bkg_transform=make_transform_val(image_output_size=224),
                             validation=True,
                             )

    sampler = torch.utils.data.RandomSampler(dataset,
                                             replacement=True,
                                             num_samples=args.iters_per_epoch * args.batch_size)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             sampler=sampler,
                                             # num_workers=10,
                                             )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=4,
                                                 shuffle=True,
                                                 )

    for e in range(args.epochs):
        loss_avg = 0.0
        loss_avg_val = 0.0

        # Training cycle
        model.train()
        for i, (image, bkg) in enumerate(tqdm(dataloader)):
            # image = image.cuda(non_blocking=True)
            # bkg = bkg.cuda(non_blocking=True)

            output = model(image)
            loss = loss_fn(output, bkg)

            optim.zero_grad()

            loss.backward()
            loss_avg += loss.item()

            optim.step()
            scheduler.step()

        # Validation cycle
        model.eval()
        for image, bkg, min_, img_mean, img_std in val_dataloader:
            # image = image.cuda(non_blocking=True)
            # bkg = bkg.cuda(non_blocking=True)
            output = model(image)

            for b in range(image.shape[0]):
                output[b] = torch.exp(output[b] * img_std[b] / 0.1 + img_mean[b]) + min_[b] - 1e-6
                bkg[b] = torch.exp(bkg[b] * img_std[b] / 0.1 + img_mean[b]) + min_[b] - 1e-6

            loss_val = loss_fn_val(output, bkg)

            loss_avg_val += loss_val.item()

        # Plotting
        utils.plot_batch(image, output, bkg, e, num_of_images=2)

        print(f"Epoch : {e}; Loss : {loss_avg / len(dataloader)}; Val Loss : {loss_avg_val / len(val_dataloader)}")

        if args.save_model:
            torch.save(model.state_dict(), "./model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train GraXpertAI background extraction",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--initial_lr", type=float, default=1e-5)
    parser.add_argument("--final_lr", type=float, default=5e-6)
    parser.add_argument("--iters_per_epoch", type=int, default=600)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--dataset_folder", type=str, default="./dataset/train/")
    parser.add_argument("--validation_folder", type=str, default="./dataset/val")

    args = parser.parse_args()

    train(args)
