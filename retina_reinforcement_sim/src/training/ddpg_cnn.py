import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ddpg_base import DdpgBase


class DdpgCnn(DdpgBase):
    """DDPG agent using a CNN."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, num_images, num_actions,
                 reward_scale):
        """Initialise agent.

        Args:
            num_images (int): Number of images in the observation
            num_actions (int): Number of possible actions
        """
        DdpgBase.__init__(self, memory_capacity, batch_size, noise_function,
                      init_noise, final_noise, exploration_len,
                      ActorCNN, [num_images, num_actions],
                      CriticCNN, [num_images, num_actions], reward_scale)

        self.num_images = num_images

        # Create transform function to prepare an image for the models
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((64, 64)),
            T.ToTensor()
        ])

    def interpet(self, obs):
        state_tensor = torch.ones((self.num_images, 64, 64)).to(self.device)
        for i in range(self.num_images):
            state_tensor[i] = self.transform(obs[i])
        return state_tensor


class ActorCNN(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        # Create network
        super(ActorCNN, self).__init__()
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


class CriticCNN(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        # Create network
        super(CriticCNN, self).__init__()
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
