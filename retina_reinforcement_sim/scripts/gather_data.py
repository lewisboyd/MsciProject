#!/usr/bin/env python

import os

import torch

from training import DdpgCnn, DdpgRetina, OrnsteinUhlenbeckActionNoise


class Pendulum:
    """Pendulum environment with action repeats returing images and state."""

    def __init__(self):
        """Initialise environment."""
        self.env = gym.make('Pendulum-v0')

    def reset(self):
        """Reset the environment."""
        self.env.reset()

    def step(self, action):
        """Execute the action returning the new observation and state."""
        action = action * 2
        _, _, _, _ = self.env.step(action)
        img1 = self.env.render(mode='rgb_array')
        _, _, _, _ = self.env.step(action)
        img2 = self.env.render(mode='rgb_array')
        new_state, _, _, _ = self.env.step(action)
        img3 = self.env.render(mode='rgb_array')
        new_obs = np.stack((img1, img2, img3))
        return new_obs, new_state

    def close(self):
        """Close the environment."""
        self.env.close()


if __name__ == '__main__':
    # Create folder to save data
    DATA_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/data/")
    if not os.path.isdir(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Create agents purely for processing images
    agentCnn = DdpgCnn(0, 0, None, 1, 1, 1, 3, 1, 1)
    agentRetina = DdpgRetina(0, 0, None, 1, 1, 1, 3, 1, 1, (500, 500))

    # Create environment
    env = Pendulum()

    # Create function for exploration
    noise_function = OrnsteinUhlenbeckActionNoise(1)

    # Create tensors for storing data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = torch.ones((100000, 3, 64, 64), dtype=torch.float, device=device)
    retina_images = torch.ones((100000, 3, 64, 64), dtype=torch.float,
                               device=device)
    targets = torch.ones((100000, 3), dtype=torch.float, device=device)

    # Populate tensors
    for ep in range(500):
        # Reset noise function and environment
        noise_function.reset()
        env.reset()

        for step in range(200):
            # Step through environment
            action = torch.tensor(self.noise_function()).clamp(-1, 1)
            obs, state = env.step(action)

            # Save processed observations and low state descriptor
            index = (ep * 200) + step
            images[index] = agentCnn.interpet(obs)
            retina_images[index] = agentRetina.interpet(obs)
            targets[index] = state.to(device)

    # Save data
    torch.save(images, DATA_FOLDER + "images")
    torch.save(retina_images, DATA_FOLDER + "retina_images")
    torch.save(targets, DATA_FOLDER + "targets")

    env.close()
