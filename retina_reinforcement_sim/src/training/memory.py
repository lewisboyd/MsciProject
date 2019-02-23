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
            capacity (int) : max number of experiences to store

        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        """Save an experience overwritting an old experience if memory is full.

        Args:
            state (object) : The state the action executed in
            action (list of float) : The action values
            next_state (object) : Next state after action executed
            reward (float) : Reward given for executing the action
            done (boolean) : If the episode is now finished

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
            batch_size (int) : Number of experiences to sample

        Returns:
            list of Experiences : randomly choosen list of experiences

        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the length of the ReplayMemory."""
        return len(self.memory)
