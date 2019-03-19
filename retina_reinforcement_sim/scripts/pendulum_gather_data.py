#!/usr/bin/env python

import os

import gym
import numpy as np
import torch

from training import (ImagePreprocessor, RetinaPreprocessor,
                      OrnsteinUhlenbeckActionNoise)


class Pendulum:
    """Pendulum environment with action repeats returing images and state."""

    def __init__(self):
        """Initialise environment."""
        self.env = gym.make('Pendulum-v0')

    def reset(self):
        """Reset the environment."""
        self.env.reset()
        obs, state, _ = self.step([0])
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
    # Variables
    SIZE = 100000

    # Create folder to save data
    DATA_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/data/")
    if not os.path.isdir(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Create preprocessors
    imagePreprocessor = ImagePreprocessor(3)
    # retinaPreprocessor = RetinaPreprocessor(3, 500, 500)

    # Create environment
    env = Pendulum()

    # Create function for exploration
    noise_function = OrnsteinUhlenbeckActionNoise(1)

    # Create tensors for storing data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = torch.ones((SIZE, 3, 64, 64), dtype=torch.float, device=device)
    next_images = torch.ones((SIZE, 3, 64, 64), dtype=torch.float,
                             device=device)
    # retina_images = torch.ones((SIZE, 3, 64, 64), dtype=torch.float,
    #                            device=device)
    # next_retina_images = torch.ones((SIZE, 3, 64, 64), dtype=torch.float,
    #                                 device = device)
    states = torch.ones((SIZE, 3), dtype=torch.float, device=device)
    next_states = torch.ones((SIZE, 3), dtype=torch.float, device=device)
    actions = torch.ones((SIZE, 1), dtype=torch.float, device=device)
    rewards = torch.ones((SIZE, 1), dtype=torch.float, device=device)
    dones = torch.ones((SIZE, 1), dtype=torch.float, device=device)

    # Populate tensors
    index = -1
    while index < SIZE:
        # Reset noise function and environment
        noise_function.reset()
        obs, state = env.reset()
        image = imagePreprocessor(obs).to(device)
        # retina_image = retinaPreprocessor(obs).to(device)

        for step in range(200):
            index += 1
            if index == SIZE:
                break

            # Step through environment
            action = torch.tensor(noise_function(), dtype=torch.float,
                                  device=device).clamp(-1, 1)
            next_obs, next_state, reward = env.step(action)
            done = 0 if step == 199 else 1

            # Get next image and retina image
            next_image = imagePreprocessor(next_obs).to(device)
            # next_retina_image = retinaPreprocessor(next_obs).to(device)

            # Normalise state to between -1, 1
            state[2] = state[2] / 8

            # Save processed observations and state descriptor
            images[index] = image
            next_images[index] = next_image
            # retina_images[index] = retina_image
            # next_retina_images[index] = next_retina_image
            states[index] = torch.tensor(state, device=device)
            next_states[index] = torch.tensor(next_state, device=device)
            actions[index] = action
            rewards[index] = torch.tensor(reward, device=device)
            dones[index] = torch.tensor(done, device=device)

            # Update current image, retina_image and state
            image = next_image
            # retina_image = next_retina_image
            state = next_state

    # Save data
    torch.save(images, DATA_FOLDER + "images")
    torch.save(next_images, DATA_FOLDER + "next_images")
    # torch.save(retina_images, DATA_FOLDER + "retina_images")
    # torch.save(next_retina_images, DATA_FOLDER + "next_retina_images")
    torch.save(states, DATA_FOLDER + "states")
    torch.save(next_states, DATA_FOLDER + "next_states")
    torch.save(actions, DATA_FOLDER + "actions")
    torch.save(rewards, DATA_FOLDER + "rewards")
    torch.save(dones, DATA_FOLDER + "dones")

    # Close environment
    env.close()
