import gym
import numpy as np


class PendulumLow:
    """Pendulum environment."""

    def __init__(self):
        """Initialise environment."""
        self.env = gym.make('Pendulum-v0')

    def reset(self):
        """Reset the environment returning the new initial state."""
        state = self.env.reset()
        self.env.render()
        return state

    def step(self, action):
        """Execute the action returning the new state and reward."""
        action = action * 2
        new_state, reward, _, _ = self.env.step(action)
        self.env.render()
        return new_state, reward

    def close(self):
        """Close the environment."""
        self.env.close()


class PendulumPixel:
    """Pendulum environment with action repeats returning image observations."""

    def __init__(self):
        """Initialise environment."""
        self.env = gym.make('Pendulum-v0')

    def reset(self):
        """Reset the environment returning the new initial state."""
        self.env.reset()
        img1 = self.env.render(mode='rgb_array')
        self.env.step([0])
        img2 = self.env.render(mode='rgb_array')
        self.env.step([0])
        img3 = self.env.render(mode='rgb_array')
        return np.stack((img1, img2, img3))

    def step(self, action):
        """Execute the action returning the new state and reward."""
        action = action * 2
        _, reward1, _, _ = self.env.step(action)
        img1 = self.env.render(mode='rgb_array')
        _, reward2, _, _ = self.env.step(action)
        img2 = self.env.render(mode='rgb_array')
        _, reward3, _, _ = self.env.step(action)
        img3 = self.env.render(mode='rgb_array')
        new_obs = np.stack((img1, img2, img3))
        reward = reward1 + reward2 + reward3
        return new_obs, reward

    def close(self):
        """Close the environment."""
        self.env.close()
