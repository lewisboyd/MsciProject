from model import Actor, Critic
from memory import ReplayMemory
from noise import NormalActionNoise
import torch
import torchvision.transforms as T


class DDPG:
    """Class responsible for network optimisation and deciding action."""

    def __init__(self, memory_capacity=1000, batch_size=32,
                 noise_function=NormalActionNoise(0, 0.8, 3), init_noise=1.0,
                 final_noise=0.0, exploration_len=1000):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int) : Maximum capacity of the memory
            batch_size (int) : Sample size from memory when optimising
            critic_loss (object) : Loss function for the critic

        """
        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 0.001)

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 0.001)

        device = torch.device("cuda:0")
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

        self.memory = ReplayMemory(memory_capacity)
        self.noise_function = noise_function
        self.noise = init_noise
        self.final_noise = final_noise
        self.noise_decay = (init_noise - final_noise) / exploration_len
        self.batch_size = batch_size
        self.tau = 0.001
        self.discount = 0.99
        self.iter = 0

    def get_exploration_action(self, state):
        """Get an action from the actor with added noise if in exploration.

        Args:
            state (object) : BGR cortical image

        Returns:
            list of float : Action Values with added noise

        """
        noise_values = torch.tensor(self.noise_function())
        if self.noise > self.final_noise:
            self.noise -= self.noise_decay
            if self.noise < self.final_noise:
                self.noise = self.final_noise

        action_values = (self.actor(T.ToTensor()(state).cuda().unsqueeze(0)
                                    .float()).squeeze().cpu().detach())
        noisy_action = action_values + self.noise * noise_values.float()
        return noisy_action.clamp(-1, 1).numpy()

    def optimise(self):
        """Sample a random batch then optimise critic and actor."""
        # Sample memory
        if len(self.memory) < self.batch_size:
            return
        sample = self.memory.sample(self.batch_size)
        batch = ReplayMemory.Experience(*zip(*sample))
        state_batch = (torch.tensor(batch.state).cuda().permute(0, 3, 1, 2)
                       .float() / 255)
        action_batch = torch.tensor(batch.action).cuda().float()
        reward_batch = torch.tensor(batch.reward).cuda().float()
        next_state_batch = (torch.tensor(batch.next_state).cuda()
                            .permute(0, 3, 1, 2).float() / 255)
        final_state_batch = (
            1 - torch.tensor(batch.final_state)).cuda().float()

        # Optimise critic
        next_q = self.critic_target(
            next_state_batch, self.actor_target(next_state_batch))
        expected_q = reward_batch + self.discount * final_state_batch * next_q
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
            path (string) : Path to folder to save models in
            episode (int) : Will be appended to filename

        """
        torch.save(self.actor.state_dict(), path + "actor_" + str(episode))
        torch.save(self.critic.state_dict(), path + "critic_" + str(episode))

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau)
                                    + param.data * self.tau)
