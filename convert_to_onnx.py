import torch
from models import VisionTransformer
from heads import ImageHead


def main():
    model = VisionTransformer(backbone_size="base", head_layer=ImageHead)

    state_dict = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    img = torch.randn(1, 3, 224, 224)

    onnx_model = torch.onnx.dynamo_export(model, img)
    onnx_model.save("model.onnx")
    # torch.onnx.export(model, img, "model.onnx", input_names=["gen_input_image"])


if __name__ == "__main__":
    main()