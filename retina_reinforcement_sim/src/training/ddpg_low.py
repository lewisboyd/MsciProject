import torch
import torch.nn as nn
from ddpg import DDPG
import torch.nn.functional as F


class Actor(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, state_dim, action_dim):
        """Initialise layers.

        Args:
            state_dim (int) : Number of state inputs
            action_dim (int) : Number of action ouputs

        """
        super(Actor, self).__init__()
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


class Critic(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, state_dim, action_dim):
        """Initialise layers.

        Args:
            state_dim (int) : Number of state inputs
            action_dim (int) : Number of action ouputs

        """
        super(Critic, self).__init__()
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


class DDPGLow(DDPG):
    """Class responsible for creating tensors before pushing to memory."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, state_dim, action_dim):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int) : Maximum capacity of the memory
            batch_size (int) : Sample size from memory when optimising
            noise_function (object) : Function to generate random additive
                                      noise
            init_noise (double) : Initial amount of noise to be added
            final_noise (double) : Final amount of noise to be added
            exploration_len (int) : Number of steps to decay noise over
            state_dim (int) : Number of state inputs
            action_dim (int) : Number of action ouputs

        """
        DDPG.__init__(self, memory_capacity, batch_size, noise_function,
                      init_noise, final_noise, exploration_len,
                      Actor, [state_dim, action_dim], Critic,
                      [state_dim, action_dim])

    def state_to_tensor(self, state):
        """Convert the state to a tensor ready to be used and saved.

        Args:
            state (numpy array) : The array of observations.

        Returns:
            state_tensor (float tensor) : Observations as tensor.

        """
        return torch.tensor(state, dtype=torch.float).to(self.device)

    def reward_to_tensor(self, reward):
        """Convert the reward to a tensor ready to be saved.

        Args:
            reward (float) : The reward value

        Returns:
            float tensor : Reward as a tensor

        """
        return torch.tensor([reward], dtype=torch.float).to(self.device)

    def done_to_tensor(self, done):
        """Convert the done boolean to a tensor ready to be saved.

        Args:
            done (boolean) : The done boolean

        Returns:
            float tensor : Done boolean as 1 (false) or 0 (true) tensor.

        """
        if done:
            return torch.tensor([0], dtype=torch.float).to(self.device)
        return torch.tensor([1], dtype=torch.float).to(self.device)
