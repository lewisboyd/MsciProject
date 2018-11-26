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
