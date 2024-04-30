import torch


class VisionTransformer(torch.nn.Module):
    def __init__(self,
                 backbone_size,
                 head_layer,
                 ):
        super().__init__()

        assert backbone_size in ["small", "base", "large"], \
            "Backbone size must be one of the following: small, base, large"

        self.backbone_size = backbone_size

        self.backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
        self.head = head_layer()

        print(f"Number of trainable parameters : {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, x):
        x = self.backbone.forward_features(x)
        cls_token = x["x_norm_patchtokens"]

        x = self.head(cls_token)

        return x
