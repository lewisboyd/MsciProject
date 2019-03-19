import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorSr(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, sr_net, sr_size, action_dim):
        """Initialise layers.

        Args
            state_dim (int): Number of state inputs
            action_dim (int): Number of action ouputs
        """
        # Create network
        super(ActorSr, self).__init__()
        self.sr_net = sr_net
        self.fc1 = nn.Linear(sr_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        # Initialise weights
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
        x = self.sr_net(batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).tanh()
        return x


class CriticSr(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, sr_net, sr_size, action_dim):
        """Initialise layers.

        Args:
            state_dim (int) : Number of state inputs
            action_dim (int) : Number of action ouputs

        """
        # Create network
        super(CriticSr, self).__init__()
        self.sr_net = sr_net
        self.fc1 = nn.Linear(sr_size, 400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)

        # Initialise weights
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        nn.init.uniform_(self.fc1.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        nn.init.uniform_(self.fc2.weight, -1 / math.sqrt(fan_in),
                         1 / math.sqrt(fan_in))
        nn.init.uniform_(self.fc3.weight, -0.0003, 0.0003)

    def forward(self, obs_batch, action_batch):
        """Generate Q-values for the batch of states and actions pairs.

        Args:
            state_batch (float tensor) : Batch of states
            action_batch (float tensor) : Batch of actions

        Returns:
            float tensor : Batch of Q-values

        """
        x = self.sr_net(obs_batch)
        x = F.relu(self.fc1(x))
        x = torch.cat((x, action_batch), 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorCnn(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        # Create network
        super(ActorCnn, self).__init__()
        self.conv1 = nn.Conv2d(num_images, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, num_actions)

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


class CriticCnn(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        # Create network
        super(CriticCnn, self).__init__()
        self.conv1 = nn.Conv2d(num_images, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 400)
        self.fc2 = nn.Linear(400 + num_actions, 300)
        self.fc3 = nn.Linear(300, 1)

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

    def forward(self, obs_batch, action_batch):
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


class ActorMlp(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, state_dim, action_dim):
        """Initialise layers.

        Args:
            state_dim (int) : Number of state inputs
            action_dim (int) : Number of action ouputs
        """
        # Create network
        super(ActorMlp, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, action_dim)

        # Initialise weights
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
        x = self.fc1(batch)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        x = self.fc3(x).tanh()
        return x


class CriticMlp(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, state_dim, action_dim):
        """Initialise layers.

        Args:
            state_dim (int) : Number of state inputs
            action_dim (int) : Number of action ouputs

        """
        # Create network
        super(CriticMlp, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.ln1 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)

        # Initialise weights
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
        x = self.fc1(state_batch)
        x = self.ln1(x)
        x = F.relu(x)
        x = torch.cat((x, action_batch), 1)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x
