from memory import ReplayMemory
import torch
from torch.nn import functional as F


class DDPG:
    """Class responsible for generating actions and optimizing models."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, actor, actor_args, critic,
                 critic_args):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int) : Maximum capacity of the memory
            batch_size (int) : Sample size from memory when optimising
            noise_function (object) : Function to generate random additive
                                      noise
            init_noise (double) : Initial amount of noise to be added
            final_noise (double) : Final amount of noise to be added
            exploration_len (int) : Number of steps to decay noise over
            actor (object): Actor constructor to use
            actor_args (array): Args for actor constructor
            critic (object): Critic constructor to use
            critic_args (array): Args for critic constructor

        """
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.noise_function = noise_function
        self.noise = init_noise
        self.final_noise = final_noise
        self.noise_decay = (init_noise - final_noise) / exploration_len
        self.tau = 0.001
        self.discount = 0.99
        self.device = torch.device("cuda:0")

        # Create actor networks
        self.actor = actor(*actor_args).to(self.device)
        self.actor_target = actor(*actor_args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 0.0001)

        # Create critic networks
        self.critic = critic(*critic_args).to(self.device)
        self.critic_target = critic(*critic_args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 0.001)

    def get_exploitation_action(self, state):
        """Generate action policy without noise.

        Args:
            state (float tensor): State tensor

        Returns:
            float tensor: Action policy

        """
        with torch.no_grad():
            self.actor.eval()
            action = self.actor(state.unsqueeze(0)).view(1)
            self.actor.train()
            return action

    def get_exploration_action(self, state):
        """Generate noisy action policy.

        Args:
            state (float tensor): State tensor

        Returns:
            float tensor: Noisy action policy

        """
        with torch.no_grad():
            noise_values = torch.tensor(
                self.noise_function(), device=self.device)
            # if self.noise > self.final_noise:
            #     self.noise -= self.noise_decay
            #     if self.noise < self.final_noise:
            #         self.noise = self.final_noise
            action = self.actor(state.unsqueeze(0)).view(1)
            noisy_action = action + self.noise * noise_values.float()
            return noisy_action.clamp(-1, 1)

    def optimise(self):
        """Sample a random batch then optimise critic and actor."""
        # Sample memory
        if len(self.memory) < self.batch_size:
            return None, None
        sample = self.memory.sample(self.batch_size)
        batch = ReplayMemory.Experience(*zip(*sample))
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.stack(batch.done)

        # Compute critic loss
        with torch.no_grad():
            next_q = self.critic_target(
                next_state_batch, self.actor_target(next_state_batch))
            expected_q = reward_batch + (self.discount * done_batch * next_q)
        predicted_q = self.critic.forward(state_batch, action_batch)
        critic_loss = F.mse_loss(predicted_q, expected_q)

        # Optimise critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_value_(self.critic.parameters(), 1)
        self.critic_optim.step()

        # Compute actor loss
        actor_loss = -self.critic(state_batch,
                                  self.actor(state_batch)).mean()

        # Optimise actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update
        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)

    def save(self, path, id):
        """Save the models weights.

        Args:
            path (string) : Path to folder to save models in
            id (string) : Will be appended to filename

        """
        torch.save(self.actor.state_dict(), path + "actor_" + id)
        torch.save(self.critic.state_dict(), path + "critic_" + id)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau)
                                    + param.data * self.tau)
