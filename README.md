# AstroNet
AstroNet is a repository that makes simple to train an AI model in the realm of astrophotography.

It has been developed, in particular, for the GraXpert team and the fantastic work done by Steffen has been a fondamental 
inspiration for this library.

## Model supported
| Name                  | Training           | Preprocessing                  |
|-----------------------|--------------------|--------------------------------|
| Background extraction | `train_bge.py`     | `data/prepare_fits_bge.py`     |
| Denoise               | `train_denoise.py` | `data/prepare_fits_denoise.py` |

## Train background extraction
1. ``python data/prepare_fits_bge.py --img_folder <images with gradients> --bkg_folder <gradients> --output_folder <folder in which you want the dataset to be created>``
2. ``python train_bge.py ``