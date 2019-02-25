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
        self.conv1 = nn.Conv2d(num_images, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, state_dim)

    def forward(self, batch):
        """Generate state descriptors for the batch of observations.

        Args:
            batch (float tensor) : Batch of observations

        Returns:
            float tensor : Batch of state desciptors
        """
        x = F.relu(self.conv1(batch))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch.size(0), 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).tanh()
        return x


class FeatureExtractor2(nn.Module):
    """Network for regressing low state descriptor from image sequences."""

    def __init__(self, num_images, state_dim):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            state_dim (int) : Number of low state descriptors
        """
        # Create network
        super(FeatureExtractor2, self).__init__()
        self.conv1 = nn.Conv2d(num_images, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, state_dim)

    def forward(self, batch):
        """Generate state descriptors for the batch of observations.

        Args:
            batch (float tensor) : Batch of observations

        Returns:
            float tensor : Batch of state desciptors
        """
        x = F.relu(self.conv1(batch))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch.size(0), 1024)
        x = self.fc1(x).tanh()
        return x
