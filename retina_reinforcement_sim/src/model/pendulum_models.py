import torch
import torch.nn as nn
import torch.nn.functional as F


class PendulumActor(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, feature_extractor, num_features, num_images,
                 num_actions):
        """Initialise layers.

        Args:
            feature_extractor (model) : The model to extract image features
            num_features (int) : Number of features the extractor returns
            num_images (int) : Number of RGB images
            num_actions (int) : Number of actions

        """
        super(PendulumActor, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_features = num_features
        self.num_images = num_images
        self.fc1 = nn.Linear(num_features * num_images, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
        nn.init.uniform(fc3.weight, -0.0003, 0.0003)

    def forward(self, batch):
        """Generates action policy for the batch of states.

        Args:
            batch (float tensor) : Batch of states

        Returns:
            float tensor: Batch of action policies

        """
        b = batch.size(0)
        x = batch.view(b * self.num_images, 3, 224, 224)
        x = self.feature_extractor(x)
        x = x.view(b, self.num_features * self.num_images)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).tanh()
        return x


class PendulumCritic(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, feature_extractor, num_features, num_images,
                 num_actions):
        """Initialise layers.

        Args:
            feature_extractor (model) : The model to extract image features
            num_features (int) : Number of features the extractor returns
            num_images (int) : Number of RGB images
            num_actions (int) : Number of actions

        """
        super(PendulumCritic, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_features = num_features
        self.num_images = num_images
        self.fc1 = nn.Linear(num_features * num_images + num_actions, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        nn.init.uniform(fc3.weight, -0.0003, 0.0003)

    def forward(self, state_batch, action_batch):
        """Generates Q-values for the batch of states and action pairs.

        Args:
            state_batch (float tensor) : Batch of states
            action_batch (float tensor) : Batch of actions

        Returns:
            float tensor : Batch of Q-values

        """
        b = state_batch.size(0)
        x = state_batch.view(b * self.num_images, 3, 224, 224)
        x = self.feature_extractor(state)
        x = x.view(b, self.num_features * self.num_images)
        x = torch.cat((x, action_batch), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
