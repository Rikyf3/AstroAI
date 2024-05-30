import torch
from models import UNet

def main():
    model = UNet([64, 128, 256, 512, 512, 512, 512, 512])

    state_dict = torch.load("model_denoise.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    img = torch.randn(1, 1, 256, 256)

    torch.onnx.export(model,
                      img,
                      "model.onnx",
                      input_names=["gen_input_image"],
                      output_names=["output"],
                      dynamic_axes={'gen_input_image': {0: 'batch_size'}, 'output': {0: 'batch_size'}})


if __name__ == "__main__":
    main()
