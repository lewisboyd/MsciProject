import torch
import torch.nn as nn
import torch.nn.functional as F


class PendulumActorCNN(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        super(PendulumActorCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_images, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
        nn.init.uniform_(self.fc3.weight, -0.0003, 0.0003)

    def forward(self, batch):
        """Generate action policy for the batch of states.

        Args:
            batch (float tensor) : Batch of states

        Returns:
            float tensor : Batch of action policies

        """
        x = F.relu(self.conv1(batch))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch.size(0), 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).tanh()
        return x


class PendulumCriticCNN(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        super(PendulumCriticCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_images, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512 + num_actions, 256)
        self.fc3 = nn.Linear(256, 1)
        nn.init.uniform_(self.fc3.weight, -0.0003, 0.0003)

    def forward(self, state_batch, action_batch):
        """Generate Q-values for the batch of states and actions pairs.

        Args:
            state_batch (float tensor) : Batch of states
            action_batch (float tensor) : Batch of actions

        Returns:
            float tensor : Batch of Q-values

        """
        x = F.relu(self.conv1(state_batch))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(state_batch.size(0), 1024)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, action_batch), 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PendulumActorLow(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, state_dim, action_dim):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        super(PendulumActorLow, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        nn.init.uniform_(self.fc3.weight, -0.0003, 0.0003)

    def forward(self, batch):
        """Generate action policy for the batch of states.

        Args:
            batch (float tensor) : Batch of states

        Returns:
            float tensor : Batch of action policies

        """
        x = F.relu(self.fc1(batch))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).tanh()
        return x


class PendulumCriticLow(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, state_dim, action_dim):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        super(PendulumCriticLow, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)
        nn.init.uniform_(self.fc3.weight, -0.0003, 0.0003)

    def forward(self, state_batch, action_batch):
        """Generate Q-values for the batch of states and actions pairs.

        Args:
            state_batch (float tensor) : Batch of states
            action_batch (float tensor) : Batch of actions

        Returns:
            float tensor : Batch of Q-values

        """
        x = F.relu(self.fc1(state_batch))
        x = torch.cat((x, action_batch), 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
