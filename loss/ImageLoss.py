import torch


class ImageLoss(torch.nn.Module):
    def __init__(self):
        super(ImageLoss, self).__init__()

    def forward(self, x, y):
        l1_loss = 100 * torch.mean(torch.abs(x - y))
        log_l1_loss = torch.mean(torch.log(torch.abs(x - y) * 100 + 1e-5))

        return l1_loss + log_l1_loss
