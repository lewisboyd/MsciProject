from model import PendulumActor, PendulumCritic
from memory import ReplayMemory
from noise import NormalActionNoise
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as Models


class DDPG:
    """Class responsible for generating actions and optimizing models."""

    def __init__(self, memory_capacity=1000, batch_size=32,
                 noise_function=NormalActionNoise(0, 0.2, 3), init_noise=1.0,
                 final_noise=0.02, exploration_len=200):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int) : Maximum capacity of the memory
            batch_size (int) : Sample size from memory when optimising
            noise_function (object) : Function to generate random additive noise
            init_noise (double) : Initial amount of noise to be added
            final_noise (double) : Final amount of noise to be added
            exploration_len (int) : Number of steps to decay noise over

        """
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.noise_function = noise_function
        self.noise = init_noise
        self.final_noise = final_noise
        self.noise_decay = (init_noise - final_noise) / exploration_len
        self.device = torch.device("cuda:0")
        self.tau = 0.001
        self.discount = 0.99

        # Get pretrained Resnet with fully connected layers removed
        feature_extractor = Models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(feature_extractor.children())[:-1]).to(self.device)
        # for param in feature_extractor.parameters():
        #     param.requires_grad = False
        # feature_extractor.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.actor = PendulumActor(feature_extractor, 512, 3, 1).to(self.device)
        self.actor_target = PendulumActor(feature_extractor, 512, 3, 1).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 0.0001)

        self.critic = PendulumCritic(feature_extractor, 512, 3, 1).to(self.device)
        self.critic_target = PendulumCritic(feature_extractor, 512, 3, 1).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 0.001)

    def state_to_tensor(self, state):
        """Converts the state to a normalised tensor ready to be used and saved.

        Args:
            state (numpy array) : The array of RGB images.

        Returns:
            state_tensor (float tensor) : Tensor with images resized and
                                          normalised.

        """
        state_tensor = torch.ones((3,3,224,224), device='cuda:0')
        state_tensor[0] = self.transform(x[0])
        state_tensor[1] = self.transform(x[1])
        state_tensor[2] = self.transform(x[2])
        return state_tensor

    def reward_to_tensor(self, reward):
        """Convert the reward to a tensor ready to be saved.

        Args:
            reward (float) : The reward value

        Returns:
            float tensor : Reward as a tensor

        """
        return torch.tensor([reward], dtype=torch.float)

    def done_to_tensor(self, done):
        """Convert the done boolean to a tensor ready to be saved.

        Args:
            done (boolean) : The done boolean

        Returns:
            float tensor : Done boolean as 1 (false) or 0 (true) tensor.

        """
        if done:
            return torch.tensor([0], dtype=torch.float)
        return torch.tensor([1], dtype=torch.float)

    def get_exploitation_action(self, state):
        """Generates the action policy without noise.

        Args:
            state (float tensor): Preprocessed state tensor

        Returns:
            float tensor: Action policy

        """
        with torch.no_grad():
            self.feature_extractor.eval()
            action = self.actor(state.unsqueeze(0))
            self.feature_extractor.train()
            return action.to('cpu')


    def get_exploration_action(self, state):
        """Generates a noisy action policy.

        Args:
            state (float tensor): Preprocessed state tensor

        Returns:
            float tensor: Noisy action policy

        """
        with torch.no_grad():
            noise_values = torch.tensor(
                self.noise_function(), device=self.device)
            if self.noise > self.final_noise:
                self.noise -= self.noise_decay
                if self.noise < self.final_noise:
                    self.noise = self.final_noise
            action = self.actor(state.unsqueeze(0))
            noisy_action = action + self.noise * noise_values.float()
            return noisy_action.clamp(-1, 1).to('cpu')

    def optimise(self):
        """Sample a random batch then optimise critic and actor."""
        # Sample memory
        if len(self.memory) < self.batch_size:
            return None, None
        sample = self.memory.sample(self.batch_size)
        batch = ReplayMemory.Experience(*zip(*sample))
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.stack(batch.done).to(self.device)

        # Optimise critic
        next_q = self.critic_target(
            next_state_batch, self.actor_target(next_state_batch))
        expected_q = reward_batch + self.discount * done_batch * next_q
        predicted_q = self.critic.forward(state_batch, action_batch)
        critic_loss = (expected_q - predicted_q).pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Optimise actor
        actor_loss = -self.critic(state_batch,
                                  self.actor(state_batch)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update
        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)

    def save(self, path, episode):
        """Save the models weights.

        Args:
            path(string): Path to folder to save models in
            episode(int): Will be appended to filename

        """
        torch.save(self.actor.state_dict(), path + "actor_" + str(episode))
        torch.save(self.critic.state_dict(), path + "critic_" + str(episode))

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau)
                                    + param.data * self.tau)
