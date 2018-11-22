from collections import namedtuple
import random


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
