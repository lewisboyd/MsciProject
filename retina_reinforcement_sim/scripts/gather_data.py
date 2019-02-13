#!/usr/bin/env python

import os

import torch
import numpy as np

from training import DdpgCnn, DdpgRetina, OrnsteinUhlenbeckActionNoise


class Pendulum:
    """Pendulum environment with action repeats returing images and state."""

    def __init__(self):
        """Initialise environment."""
        self.env = gym.make('Pendulum-v0')

    def reset(self):
        """Reset the environment."""
        self.env.reset()
        obs, state, _ = self.step(0)
        return obs, state

    def step(self, action):
        """Execute the action returning the new observation and state."""
        action = action * 2
        _, reward1, _, _ = self.env.step(action)
        img1 = self.env.render(mode='rgb_array')
        _, reward2, _, _ = self.env.step(action)
        img2 = self.env.render(mode='rgb_array')
        new_state, reward3, _, _ = self.env.step(action)
        img3 = self.env.render(mode='rgb_array')
        new_obs = np.stack((img1, img2, img3))
        reward = reward1 + reward2 + reward3
        return new_obs, new_state, reward

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
    next_images = torch.ones((100000, 3, 64, 64), dtype=torch.float,
                             device=device)
    retina_images = torch.ones((100000, 3, 64, 64), dtype=torch.float,
                               device=device)
    next_retina_images = torch.ones((100000, 3, 64, 64), dtype=torch.float,
                               device=device)
    states = torch.ones((100000, 3), dtype=torch.float, device=device)
    next_states = torch.ones((100000, 3), dtype=torch.float, device=device)
    actions = torch.ones((100000, 1), dtype=torch.float, device=device)
    rewards = torch.ones((100000, 1), dtype=torch.float, device=device)
    dones = torch.ones((100000, 1), dtype=torch.float, device=device)

    # Populate tensors
    for ep in range(500):
        # Reset noise function and environment
        noise_function.reset()
        obs, state = env.reset()
        image = agentCnn.interpet(obs)
        retinaImage = agentRetina.interpet(obs)

        for step in range(200):
            # Step through environment
            action = torch.tensor(self.noise_function(), dtype = torch.float,
                                  device=device).clamp(-1, 1)
            next_obs, new_state, reward = env.step(action)
            done = 0 if step == 199 else 1

            # Get next image and retina image
            next_image = agentCnn.interpet(next_obs)
            next_retina_image = agentRetina.interpet(next_obs)

            # Normalise state to between -1, 1
            state[2] = state[2] / 8

            # Save processed observations and state descriptor
            index = (ep * 200) + step
            images[index] = image
            retinaImages[index] = retinaImage
            next_images[index] = next_image
            next_retina_images[index] = next_retina_image
            states[index] = torch.tensor(state, device=device)
            next_states[index] = torch.tensor(next_state, device=device)
            actions[index] = action
            rewards[index] = torch.tensor(reward, device=device)
            dones[index] = torch.tensor(done, device=device)

            # Update current image, retina_image and state
            image = next_image
            retina_image = next_retina_image
            state = next_state

    # Shuffle data
    np.random.seed(40)
    np.random.shuffle(images.numpy())
    np.random.seed(40)
    np.random.shuffle(next_images.numpy())
    np.random.seed(40)
    np.random.shuffle(retina_images.numpy())
    np.random.seed(40)
    np.random.shuffle(next_retina_images.numpy())
    np.random.seed(40)
    np.random.shuffle(states.numpy())
    np.random.seed(40)
    np.random.shuffle(next_states.numpy())
    np.random.seed(40)
    np.random.shuffle(actions.numpy())
    np.random.seed(40)
    np.random.shuffle(rewards.numpy())
    np.random.seed(40)
    np.random.shuffle(dones.numpy())

    # Save data
    torch.save(images, DATA_FOLDER + "images")
    torch.save(next_images, DATA_FOLDER + "next_images")
    torch.save(retina_images, DATA_FOLDER + "retina_images")
    torch.save(next_retina_images, DATA_FOLDER + "next_retina_images")
    torch.save(states, DATA_FOLDER + "states")
    torch.save(next_states, DATA_FOLDER + "next_states")
    torch.save(actions, DATA_FOLDER + "actions")
    torch.save(rewards, DATA_FOLDER + "rewards")
    torch.save(done, DATA_FOLDER + "dones")

    # Close environment
    env.close()
