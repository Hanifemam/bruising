from torch import nn


class ShallowCNN(nn.Module):
    def __init__(self, in_channels, num_classes=2, conv_channels=(16, 32), kernel_size=3, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        layers, c = [], in_channels
        for out_c in conv_channels:
            layers += [nn.Conv2d(c, out_c, kernel_size, padding=padding), nn.ReLU(), nn.MaxPool2d(2)]
            c = out_c
        self.net = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c, num_classes),
        )

    def forward(self, x):
        return self.net(x)
