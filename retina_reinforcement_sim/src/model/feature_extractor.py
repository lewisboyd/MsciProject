import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Network for regressing low state descriptor from image sequences."""

    def __init__(self, num_images, state_dim):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            state_dim (int) : Number of low state descriptors
        """
        # Create network
        super(FeatureExtractor, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_images, 32, 8, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, state_dim)

    def forward(self, batch):
        """Generate state descriptors for the batch of observations.

        Args:
            batch (float tensor) : Batch of observations

        Returns:
            float tensor : Batch of state desciptors
        """
        x = self.conv1(batch)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return self.fc1(x).tanh()


class BasicBlock(nn.Module):
    """2 layer full preactivion ResNet block."""

    def __init__(self, inplanes, outplanes, stride):
        """Initialise layers.

        Args:
            inplanes (int): Number of inplanes.
            outplanes (int): Desired number of outplanes.
            stride (int): Stride for first conv layer.
        """
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.downsample = nn.Sequential()
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(outplanes)
            )

    def forward(self, x):
        """Process input tensor."""
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.downsample(x)

        return out


class ResNet10(nn.Module):
    """10 Layer ResNet (4.9M params) using full preactivion."""

    def __init__(self, state_dim):
        """Initialise layers.

        Args:
            state_dim (int): Number of values to regress
        """
        # Create network
        super(ResNet10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.block1 = BasicBlock(64, 64, 1)
        self.block2 = BasicBlock(64, 128, 2)
        self.block3 = BasicBlock(128, 256, 2)
        self.block4 = BasicBlock(256, 512, 2)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, state_dim)

    def forward(self, batch):
        """Generate predictions for batch of images.

        Args:
            batch (float tensor): Batch of images.

        Returns:
            float tensor : Batch of predictions.
        """
        x = self.conv1(batch)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x).tanh()


class ResNet6(nn.Module):
    """6 Layer ResNet (307K params) using full preactivion."""

    def __init__(self, state_dim):
        """Initialise layers.

        Args:
            state_dim (int): Number of values to regress
        """
        # Create network
        super(ResNet6, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.block1 = BasicBlock(64, 64, 1)
        self.block2 = BasicBlock(64, 128, 2)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, state_dim)

    def forward(self, batch):
        """Generate predictions for batch of images.

        Args:
            batch (float tensor): Batch of images.

        Returns:
            float tensor : Batch of predictions.
        """
        x = self.conv1(batch)

        x = self.block1(x)
        x = self.block2(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x).tanh()
