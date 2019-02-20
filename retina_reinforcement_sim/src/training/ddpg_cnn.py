import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ddpg_base import DdpgBase
from preprocessor import ImagePreprocessor


class DdpgCnn(DdpgBase):
    """DDPG agent using a CNN."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, reward_scale, num_images,
                 num_actions, preprocessor=None):
        """Initialise agent.

        Args:
            num_images (int): Number of images in the observation
            num_actions (int): Number of possible actions
        """
        actor = Actor(num_images, num_actions).cuda()
        actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)

        critic = Critic(num_images, num_actions).cuda()
        critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
                                        weight_decay=0.01)

        if preprocessor is None:
            preprocessor = ImagePreprocessor(num_images)

        DdpgBase.__init__(self, memory_capacity, batch_size, noise_function,
                          init_noise, final_noise, exploration_len,
                          reward_scale, actor, actor_optim, critic,
                          critic_optim, preprocessor)


class Actor(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        # Create network
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(num_images, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)

        # Initialise weights
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1.weight)
        nn.init.uniform_(self.conv1.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv2.weight)
        nn.init.uniform_(self.conv2.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv3.weight)
        nn.init.uniform_(self.conv3.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        nn.init.uniform_(self.fc1.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        nn.init.uniform_(self.fc2.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        nn.init.uniform_(self.fc3.weight, -0.0003, 0.0003)

    def forward(self, batch):
        """Generate action policy for the batch of observations.

        Args:
            batch (float tensor) : Batch of observations

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


class Critic(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        # Create network
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(num_images, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512 + num_actions, 256)
        self.fc3 = nn.Linear(256, 1)

        # Initialise weights
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1.weight)
        nn.init.uniform_(self.conv1.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv2.weight)
        nn.init.uniform_(self.conv2.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv3.weight)
        nn.init.uniform_(self.conv3.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        nn.init.uniform_(self.fc1.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        nn.init.uniform_(self.fc2.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        nn.init.uniform_(self.fc3.weight, -0.0003, 0.0003)

    def forward(self, state_batch, action_batch):
        """Generate Q-values for the batch of observations and actions pairs.

        Args:
            obs_batch (float tensor) : Batch of observations
            action_batch (float tensor) : Batch of actions

        Returns:
            float tensor : Batch of Q-values

        """
        x = F.relu(self.conv1(obs_batch))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(obs_batch.size(0), 1024)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, action_batch), 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
