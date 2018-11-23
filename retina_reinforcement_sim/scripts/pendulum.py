import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random
import gc


class NormalActionNoise:
    """Normally distributed noise."""

    def __init__(self, mu=0, sigma=1, actions=1):
        """Initialise parameters.

        Args:
            mu (float) : mean of distribution
            sigma (float) : spread of standard deviation
            actions (int) : number of actions to generate noise for

        """
        self.mu = mu
        self.sigma = sigma
        self.actions = actions

    def __call__(self):
        """Return normally distributed noise."""
        return np.random.normal(self.mu, self.sigma, self.actions)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim, 256)
        self.fcs2 = nn.Linear(256, 128)

        self.fca1 = nn.Linear(action_dim, 128)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2, a1), dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_lim):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.tanh(self.fc4(x))

        action = action

        return action


class ReplayMemory(object):
    """Class to store and sample experiences during training."""

    Experience = namedtuple('Experience',
                            ('state', 'action', 'next_state', 'reward',
                             'final_state'))

    def __init__(self, capacity):
        """Create empty list bounded by the capacity.

        Args:
            capacity (int) : max number of experiences to store

        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, final_state):
        """Save an experience overwritting an old experience if memory is full.

        Args:
            state (object) : The state the action executed in
            action (list of float) : The action values
            next_state (object) : Next state after action executed
            reward (float) : Reward given for executing the action

        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (
            ReplayMemory.Experience(state, action, next_state, reward,
                                    final_state))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Return a list of randomly choosen experiences.

        Args:
            batch_size (int) : Number of experiences to sample

        Returns:
            list of Experiences : randomly choosen list of experiences

        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the length of the ReplayMemory."""
        return len(self.memory)


class DDPG:
    """Class responsible for network optimisation and deciding action."""

    def __init__(self, s_dim, a_dim, a_max, memory_capacity=1000000, batch_size=128,
                 noise_function=NormalActionNoise(), init_noise=1.0,
                 final_noise=0.0, exploration_len=20000):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int) : Maximum capacity of the memory
            batch_size (int) : Sample size from memory when optimising
            critic_loss (object) : Loss function for the critic

        """
        self.actor = Actor(s_dim, a_dim, a_max)
        self.actor_target = Actor(s_dim, a_dim, a_max)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 0.001)

        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)
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

        action_values = (self.actor(torch.tensor(state).cuda()
                                    .float()).cpu().detach()) * 2
        noisy_action = action_values + self.noise * noise_values.float()
        return noisy_action.clamp(-2, 2).numpy()

    def optimise(self):
        """Sample a random batch then optimise critic and actor."""
        # Sample memory
        if len(self.memory) < self.batch_size:
            return
        sample = self.memory.sample(self.batch_size)
        batch = ReplayMemory.Experience(*zip(*sample))
        state_batch = torch.tensor(batch.state).cuda()
        action_batch = torch.tensor(batch.action).cuda()
        reward_batch = torch.tensor(batch.reward).cuda()
        next_state_batch = torch.tensor(batch.next_state).cuda()
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
        actor_loss = -(self.critic(state_batch,
                                   self.actor(state_batch))).mean()

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


env = gym.make('Pendulum-v0')

MAX_EPISODES = 1000
MAX_STEPS = 200
MAX_BUFFER = 10000
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

trainer = DDPG(S_DIM, A_DIM, A_MAX)

for _ep in range(MAX_EPISODES):
    observation = env.reset()
    ep_reward = 0
    print 'EPISODE :- ', _ep
    for r in range(MAX_STEPS):
        env.render()
        state = np.float32(observation)

        action = trainer.get_exploration_action(state)
        # if _ep%5 == 0:
        # 	# validate every 5th episode
        # 	action = trainer.get_exploitation_action(state)
        # else:
        # 	# get action based on observation, use exploration policy here
        # 	action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(action)
        ep_reward += reward

        # # dont update if this is validation
        # if _ep%50 == 0 or _ep>450:
        # 	continue

        if done:
            new_state = None
        else:
            new_state = np.float32(new_observation)
            # push this exp in ram
            trainer.memory.push(state, action, new_state, [reward], [done])

        observation = new_observation

        # perform optimization
        trainer.optimise()
        if done:
            break

    print "Total Reward : " + str(ep_reward)

    # check memory consumption and clear memory
    gc.collect()
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss)
