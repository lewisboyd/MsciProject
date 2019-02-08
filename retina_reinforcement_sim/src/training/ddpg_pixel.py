import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ddpg import DDPG


class DDPGPixel(DDPG):
    """Class responsible for creating tensors before pushing to memory."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, num_images, num_actions):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int): Maximum capacity of the memory
            batch_size (int): Sample size from memory when optimising
            noise_function (object): Function to generate random additive
                                     noise
            init_noise (double): Initial amount of noise to be added
            final_noise (double): Final amount of noise to be added
            exploration_len (int): Number of steps to decay noise over
            num_images (int): Number of images in the state
            num_actions (int): Number of possible actions

        """
        DDPG.__init__(self, memory_capacity, batch_size, noise_function,
                      init_noise, final_noise, exploration_len,
                      ActorCNN, [num_images, num_actions],
                      CriticCNN, [num_images, num_actions])

        self.num_images = num_images

        # Create transform function to prepare an image for the models
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((64, 64)),
            T.ToTensor()
        ])

    def state_to_tensor(self, state):
        """Convert the state to a normalised tensor ready to be used and saved.

        Args:
            state (numpy array) : The array images.

        Returns:
            state_tensor (float tensor) : Tensor with images resized and
                                          converted to greyscale.

        """
        state_tensor = torch.ones((self.num_images, 64, 64)).to(self.device)
        for i in range(self.num_images):
            state_tensor[i] = self.transform(state[i])
        return state_tensor

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


class ActorCNN(nn.Module):
    """Represents an Actor in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        super(ActorCNN, self).__init__()
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


class CriticCNN(nn.Module):
    """Represents a Critic in the Actor to Critic Model."""

    def __init__(self, num_images, num_actions):
        """Initialise layers.

        Args:
            num_images (int) : Number of greyscale images
            num_actions (int) : Number of actions

        """
        super(CriticCNN, self).__init__()
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
