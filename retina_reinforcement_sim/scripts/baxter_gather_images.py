#!/usr/bin/env python

import os
import time

import cv2
import numpy as np
import torch
import rospy

from training import BaxterPreprocessor, NormalActionNoise
from environment import BaxterEnvironment, Retina


def block_visible(img):
    """Check the block is sufficiently visible."""
    # Define lower and upper colour bounds
    BLUE_MIN = np.array([100, 150, 0], np.uint8)
    BLUE_MAX = np.array([140, 255, 255], np.uint8)

    # Threshold the image in the HSV colour space
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, BLUE_MIN, BLUE_MAX)

    # Check if object visible
    if cv2.countNonZero(mask) < 150:
        return False
    return True


if __name__ == '__main__':
    start = time.time()

    # Variables
    SIZE = 100000
    STATE_DIM = 2
    ACTION_DIM = 2
    MAX_EP_STEPS = 100

    # Create folders to save data
    DATA_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/image_data/")
    NORM_IMAGES = DATA_FOLDER + "images/"
    RETINA_IMAGES = DATA_FOLDER + "retina_images/"
    if not os.path.isdir(NORM_IMAGES):
        os.makedirs(NORM_IMAGES)
    if not os.path.isdir(RETINA_IMAGES):
        os.makedirs(RETINA_IMAGES)

    # Create preprocessors
    preprocessor = BaxterPreprocessor()

    # Create environment
    rospy.init_node('data_gathering_process')
    env = BaxterEnvironment((1280, 800))
    retina = Retina(1280, 800)
    rospy.on_shutdown(env.close)

    # Create function for exploration
    noise_function = NormalActionNoise(actions=ACTION_DIM)

    # Create tensors for storing data
    states = torch.ones((SIZE, STATE_DIM), dtype=torch.float)

    timestep_ep = 0
    done = True
    state = None
    image = None
    index = 0
    while index < SIZE:
        # If episode finished start a new one and reset noise function
        if done:
            state = preprocessor(env.reset())
            image = env.image
            noise_function.reset()
            timestep_ep = 0

        retina_image = retina.sample(image)
        # If not visible through retina then skip data and reset
        if not block_visible(retina_image):
            done = True
            continue

        # Save images and state to be extracted from image
        cv2.imwrite(NORM_IMAGES + "img" + str(index) + ".png",
                    cv2.resize(image, (468, 246),
                               interpolation=cv2.INTER_AREA))
        cv2.imwrite(RETINA_IMAGES + "img" + str(index) + ".png", retina_image)
        states[index] = state[:2]

        # Step through environment using noise function
        next_obs, _, done = env.step(np.clip(noise_function(), -1, 1))
        done = done or (timestep_ep == MAX_EP_STEPS)
        next_state = preprocessor(next_obs)

        # Advance state, image and indices
        state = next_state
        image = env.image
        timestep_ep = timestep_ep + 1
        index = index + 1

    # Save corresponding states
    torch.save(states, DATA_FOLDER + "states")

    # Close environment
    env.close()

    end = time.time()
    mins = (end - start) / 60
    print "Gathered %d state-image pairs in %d hours %d minutes" % (
        SIZE, mins / 60, mins % 60)
