import torch


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Downsample, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding="same", stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding="same", stride=1)
        self.act1 = torch.nn.LeakyReLU(0.2)
        self.act2 = torch.nn.LeakyReLU(0.2)
        self.batch1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.batch2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.pool = torch.nn.AvgPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x_prime = self.act2(x)
        x = self.pool(x_prime)

        return x, x_prime


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Upsample, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding="same", stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding="same", stride=1)
        self.act1 = torch.nn.LeakyReLU(0.2)
        self.act2 = torch.nn.LeakyReLU(0.2)
        self.batch1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.batch2 = torch.nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x, x_conc):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.concat([x, x_conc], dim=1)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.act2(x)

        return x


class UNet(torch.nn.Module):
    def __init__(self, layer_sizes, image_channels=3):
        super(UNet, self).__init__()

        self.layer_sizes = layer_sizes
        self.down_layers = torch.nn.ModuleList()
        self.up_layers = torch.nn.ModuleList()

        # Initialize layers
        for index in range(len(self.layer_sizes)):
            in_channels = image_channels if index == 0 else self.layer_sizes[index - 1]
            out_channels = self.layer_sizes[index]

            self.down_layers.append(Downsample(in_channels=in_channels, out_channels=out_channels))
            self.up_layers.insert(0, Upsample(in_channels=2*out_channels, out_channels=in_channels))

    def forward(self, x):
        residuals = []

        for layer in self.down_layers:
            x, x_prime = layer(x)
            residuals.append(x_prime)

        residuals = reversed(residuals)

        for layer, residual in zip(self.up_layers, residuals):
            x = layer(x, residual)

        return x
