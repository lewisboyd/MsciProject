import numpy as np


class NormalActionNoise:
    """Normally distributed noise."""

    def __init__(self, mu=0, sigma=0.2, actions=3):
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


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def __call__(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
