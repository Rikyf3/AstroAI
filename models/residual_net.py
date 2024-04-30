import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(64, 64, [3, 3], stride=1, padding="same")
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, [3,3], stride=1, padding="same")
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.LeakyReLU()

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = self.relu1(r)
        r = self.conv2(r)
        r = self.bn2(r)
        r = self.relu2(r)

        return x + r


class ResNet(torch.nn.Module):
    def __init__(self, depth=20):
        super(ResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, [1, 1], stride=1, padding="same")
        self.blocks = torch.nn.ModuleList([ResidualBlock() for _ in range(depth)])
        self.conv2 = torch.nn.Conv2d(64, 3, [1, 1], stride=1, padding="same")

    def forward(self, x):
        x = self.conv1(x)

        for blk in self.blocks:
            x = blk(x)

        return self.conv2(x)
