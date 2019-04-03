#!/usr/bin/env python

import os
import time

import torch
import rospy

from training import (BaxterPreprocessor, NormalActionNoise)
from environment import BaxterEnvironment


if __name__ == '__main__':
    start = time.time()

    # Variables
    SIZE = 10000
    STATE_DIM = 2
    ACTION_DIM = 2
    MAX_EP_STEPS = SIZE / 100

    # Create folder to save data
    DATA_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/images/")
    if not os.path.isdir(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Create preprocessors
    preprocessor = BaxterPreprocessor()

    # Create environment
    rospy.init_node('data_gathering_process')
    env = BaxterEnvironment()
    rospy.on_shutdown(env.close)

    # Create function for exploration
    noise_function = NOISE_FUNCTION = NormalActionNoise(actions=ACTION_DIM)

    # Create tensors for storing data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    states = torch.ones((SIZE, STATE_DIM), dtype=torch.float, device=device)
    images = torch.ones((SIZE, 3, 256, 160), dtype=torch.float, device=device)

    # Populate tensors
    index = 0
    timestep_ep = 0
    done = False
    state = preprocessor(env.reset()).to(device)
    image = env.image
    while index < SIZE:
        # If episode finished start and new one and reset noise function
        if done:
            state = preprocessor(env.reset()).to(device)
            image = env.image
            noise_function.reset()
            timestep_ep = 0

        # Step through environment using noise function
        next_obs, _, done = env.step(np.clip(noise_function(), -1, 1))
        done = done or (timestep_ep == MAX_EP_STEPS)
        next_state = preprocessor(next_obs).to(device)

        # Add data to tensors
        states[index] = state
        images[index] = torch.from_numpy(image).permute(2, 0, 1).to(device)

        # Advance state and indices
        state = next_state
        timestep_ep = timestep_ep + 1
        index = index + 1

    # Save data
    torch.save(states, DATA_FOLDER + "states")
    torch.save(images, DATA_FOLDER + "images")

    # Close environment
    env.close()

    end = time.time()
    mins = (end - start) / 60
    print "Gathered %d state-image pairs in %d hours %d minutes" % (
        SIZE, mins / 60, mins % 60)
