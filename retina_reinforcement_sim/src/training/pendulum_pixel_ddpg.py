from model import PendulumActorCNN, PendulumCriticCNN
from ddpg import DDPG
import torch
import torchvision.transforms as T


class PendulumPixelDDPG(DDPG):
    """Class responsible for creating tensors before pushing to memory."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int) : Maximum capacity of the memory
            batch_size (int) : Sample size from memory when optimising
            noise_function (object) : Function to generate random additive
                                      noise
            init_noise (double) : Initial amount of noise to be added
            final_noise (double) : Final amount of noise to be added
            exploration_len (int) : Number of steps to decay noise over

        """
        DDPG.__init__(self, memory_capacity, batch_size, noise_function,
                      init_noise, final_noise, exploration_len,
                      PendulumActorCNN, [3, 1], PendulumCriticCNN, [3, 1])

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
            state (numpy array) : The array of RGB images.

        Returns:
            state_tensor (float tensor) : Tensor with images resized and
                                          converted to greyscale.

        """
        state_tensor = torch.ones((3, 64, 64)).to(self.device)
        state_tensor[0] = self.transform(state[0])
        state_tensor[1] = self.transform(state[1])
        state_tensor[2] = self.transform(state[2])
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
