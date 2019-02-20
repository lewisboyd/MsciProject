import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpg_base import DdpgBase
from preprocessor import Preprocessor


class DdpgMlp(DdpgBase):
    """DDPG agent using low dimensional state descriptor."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, reward_scale, state_dim,
                 action_dim):
        """Initialise agent.

        Args:
            state_dim (int) : Number of state inputs
            action_dim (int) : Number of action ouputs
        """
        actor = Actor(state_dim, action_dim).cuda()
        actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)

        critic = Critic(state_dim, action_dim).cuda()
        critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
                                        weight_decay=0.01)

        DdpgBase.__init__(self, memory_capacity, batch_size, noise_function,
                          init_noise, final_noise, exploration_len,
                          reward_scale, actor, actor_optim, critic,
                          critic_optim, Preprocessor())


class Actor(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, state_dim, action_dim):
        """Initialise layers.

        Args:
            state_dim (int) : Number of state inputs
            action_dim (int) : Number of action ouputs
        """
        # Create network
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
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


class Critic(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, state_dim, action_dim):
        """Initialise layers.

        Args:
            state_dim (int) : Number of state inputs
            action_dim (int) : Number of action ouputs

        """
        # Create network
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
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
