from .vision_transformer import VisionTransformer
from .unet import UNet
from .cgnet import CascadedGaze
from .nafnet import NAFNet
from .deconvnn import DeconvNN


def unet(config):
    return UNet(layer_sizes=config.model.layer_sizes, image_channels=config.model.image_channels)