from collections import namedtuple
import random


class ReplayMemory(object):
    """Class to store and sample experiences during training."""

    Experience = namedtuple('Experience',
                            ('state', 'action', 'next_state', 'reward',
                             'done'))

    def __init__(self, capacity):
        """Create empty list bounded by the capacity.

        Args:
            capacity (int): Maximum number of experiences to store

        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        """Save an experience overwritting an old experience if memory is full.

        Args:
            state (tensor): The state the action executed in
            action (tensor): The action values
            next_state (tensor): Next state after action executed
            reward (tensor): Reward given for executing the action
            done (tensor): 0 for episode finished, 1 for still in progress

        """
        if len(self.memory) < self.capacity:
            experience = ReplayMemory.Experience(
                state, action, next_state, reward, done)
            self.memory.append(experience)
        else:
            # Use copy_ instead of creating new experience due to pytorch
            # not correctly freeing the GPU memory (5x slower but no leak)
            self.memory[self.position].state.copy_(state)
            self.memory[self.position].action.copy_(action)
            self.memory[self.position].next_state.copy_(next_state)
            self.memory[self.position].reward.copy_(reward)
            self.memory[self.position].done.copy_(done)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Return a list of randomly choosen experiences.

        Args:
            batch_size (int): Number of experiences to sample

        Returns:
            list of Experiences: randomly choosen list of experiences

        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the length of the ReplayMemory."""
        return len(self.memory)


class HerMemory(object):
    """Class to store and sample experiences with goals during training."""

    Experience = namedtuple('Experience',
                            ('state', 'action', 'next_state', 'reward',
                             'done'))

    def __init__(self, capacity):
        """Create empty list bounded by the capacity.

        Args:
            capacity (int): Maximum number of experiences to store

        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, states, actions, next_states, rewards, dones, goal):
        """Save an episode of experiences with desired goal.

        Args:
            state (tensor): The states the action executed in
            action (tensor): The actions values
            next_state (tensor): Next states after actions executed
            reward (tensor): Rewards given for executing the actions
            done (tensor): 0 for episode finished, 1 for still in progress
            goal (tensor): Goal state

        """
        for i in range(len(states)):
            state = torch.cat([states[i], goal])
            action = actions[i]
            next_state = torch.cat([next_states[i], goal])
            reward = rewards[i]
            done = dones[i]
            if len(self.memory) < self.capacity:
                experience = ReplayMemory.Experience(
                    state, action, next_state, reward, done)
                self.memory.append(experience)
            else:
                # Use copy_ instead of creating new experience due to pytorch
                # not correctly freeing the GPU memory (5x slower but no leak)
                self.memory[self.position].state.copy_(state)
                self.memory[self.position].action.copy_(action)
                self.memory[self.position].next_state.copy_(next_state)
                self.memory[self.position].reward.copy_(reward)
                self.memory[self.position].done.copy_(done)
                self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Returns batch_size number of samples from the replay memory
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the length of the ReplayMemory."""
        return len(self.memory)
