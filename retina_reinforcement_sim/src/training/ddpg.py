from model import Actor, Critic
from memory import ReplayMemory
from noise import NormalActionNoise
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as Models


class DDPG:
    """Class responsible for network optimisation and deciding action."""

    def __init__(self, memory_capacity=1000, batch_size=200,
                 noise_function=NormalActionNoise(0, 0.2, 3), init_noise=1.0,
                 final_noise=0.02, exploration_len=20000):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int) : Maximum capacity of the memory
            batch_size (int) : Sample size from memory when optimising
            critic_loss (object) : Loss function for the critic

        """
        feature_extractor = Models.resnet18(pretrained=True)
        feature_extractor = nn.Sequential(
            *list(feature_extractor.children())[:-1])
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.actor = Actor(feature_extractor, 512)
        self.actor_target = Actor(feature_extractor, 512)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 0.001)

        self.critic = Critic(feature_extractor, 512)
        self.critic_target = Critic(feature_extractor, 512)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 0.001)

        self.device = torch.device("cuda:0")
        feature_extractor.to(self.device)
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.memory = ReplayMemory(memory_capacity)
        self.noise_function = noise_function
        self.noise = init_noise
        self.final_noise = final_noise
        self.noise_decay = (init_noise - final_noise) / exploration_len
        self.batch_size = batch_size
        self.tau = 0.001
        self.discount = 0.99
        self.iter = 0

    def interpret(self, state, reward, done):
        """Convert to tensors ready to be saved to memory.

        Args:
            state (numpy array) : RGB image
            reward (float) : The reward value
            done (boolean) : The done boolean

        Returns:
            state_tensor (float tensor) : Image as tensor on same device as
                                          networks.
            reward_tensor (float tensor) : Reward as tensor on same device as
                                           networks.
            done_tensor (float tensor) : Done boolean as 1 (false) or 0 (true)
                                         on same device as networks.

        """
        return (self.state_to_tensor(state), self.reward_to_tensor(reward),
                self.done_to_tensor(done))

    def state_to_tensor(self, state):
        """Convert the state to a tensor ready to be saved to memory.

        Args:
            state (numpy array) : RGB image

        Returns:
            float tensor : Image as tensor on same device as networks.

        """
        return self.transform(state).unsqueeze(0).to(self.device)

    def reward_to_tensor(self, reward):
        """Convert the reward to a tensor ready to be saved to memory.

        Args:
            reward (float) : The reward value

        Returns:
            float tensor : Reward as tensor on same device as networks

        """
        return torch.tensor([[reward]], device=self.device, dtype=torch.float)

    def done_to_tensor(self, done):
        """Convert the done boolean to a tensor ready to be saved to memory.

        Args:
            done (boolean) : The done boolean

        Returns:
            float tensor : Done boolean as 1 (false) or 0 (true) on same
                           device as networks.

        """
        if done:
            return torch.tensor([[0]], device=self.device, dtype=torch.float)
        return torch.tensor([[1]], device=self.device, dtype=torch.float)

    def get_exploration_action(self, state):
        """Get an action from the actor with added noise if in exploration.

        Args:
            state(float tensor): RGB cortical image

        Returns:
            float tensor: Action with added noise

        """
        with torch.no_grad():
            noise_values = torch.tensor(
                self.noise_function(), device=self.device)
            if self.noise > self.final_noise:
                self.noise -= self.noise_decay
                if self.noise < self.final_noise:
                    self.noise = self.final_noise
            action = self.actor(state)
            noisy_action = action + self.noise * noise_values.float()
            return noisy_action.clamp(-1, 1)

    def optimise(self):
        """Sample a random batch then optimise critic and actor."""
        # Sample memory
        if len(self.memory) < self.batch_size:
            return
        sample = self.memory.sample(self.batch_size)
        batch = ReplayMemory.Experience(*zip(*sample))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

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

        if self.iter % 100 == 0:
            print('Iteration :- ', self.iter, ' Loss_actor :- ',
                  actor_loss.data.cpu().numpy(), ' Loss_critic :- ',
                  critic_loss.data.cpu().numpy())
        self.iter += 1

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
