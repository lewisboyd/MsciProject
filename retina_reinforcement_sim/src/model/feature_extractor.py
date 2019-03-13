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
