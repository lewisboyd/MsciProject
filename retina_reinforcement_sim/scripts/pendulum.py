#!/usr/bin/env python

import os
from environment import PendulumPixel, PendulumLow
from training import DdpgMlp, DdpgCnn, DdpgRetina, OrnsteinUhlenbeckActionNoise


if __name__ == '__main__':
    # Training variables
    INIT_EXPLORE = 500
    MAX_EPISODES_PIXEL = 2000
    MAX_EPISODES_LOW = 400
    MAX_STEPS = 200
    EVAL_FREQ_LOW = 10

    # Agent variables
    REPLAY_SIZE = 100000
    NOISE_FUNCTION = OrnsteinUhlenbeckActionNoise(1)
    REWARD_SCALE = 0.1
    INIT_NOISE = 1
    FINAL_NOISE = 0.02
    EXPLORATION_LEN_PIXEL = 100000
    EXPLORATION_LEN_LOW = 10000
    ACTION_DIM = 1

    # Agent variables low state
    STATE_DIM_LOW = 3
    BATCH_SIZE_LOW = 64

    # Agent variables image
    NUM_IMAGES = 3
    BATCH_SIZE_IMAGE = 16
    IMAGE_SIZE = (500, 500)


    # Save paths
    MODEL_FOLDER_NORMAL = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/state_dicts/normal/")
    MODEL_FOLDER_PIXEL = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/state_dicts/pixel/")
    MODEL_FOLDER_RETINA = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/state_dicts/retina/")
    DATA_FOLDER_NORMAL = (os.path.dirname(os.path.realpath(__file__))
                          + "/pendulum/results/low_dim/")
    DATA_FOLDER_PIXEL = (os.path.dirname(os.path.realpath(__file__))
                         + "/pendulum/results/pixel/")
    DATA_FOLDER_RETINA = (os.path.dirname(os.path.realpath(__file__))
                          + "/pendulum/results/retina/")

    # Train DDPG agent on pendulum low state dimension
    # agent = DdpgMlp(REPLAY_SIZE, BATCH_SIZE_LOW, NOISE_FUNCTION, INIT_NOISE,
    #                 FINAL_NOISE, EXPLORATION_LEN_LOW, STATE_DIM_LOW,
    #                 ACTION_DIM, REWARD_SCALE)
    # environment = PendulumLow()
    # agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
    #             MODEL_FOLDER_NORMAL, DATA_FOLDER_NORMAL,
    #             eval_freq=EVAL_FREQ_LOW)

    # Create Pendulum environment returning 3 images
    environment = PendulumPixel(False)

    # Train DDPG agent using CNN
    agent = DdpgCnn(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION,
                    INIT_NOISE, FINAL_NOISE, EXPLORATION_LEN_PIXEL, NUM_IMAGES,
                    ACTION_DIM, REWARD_SCALE)
    agent.train(environment, INIT_EXPLORE, MAX_EPISODES_PIXEL, MAX_STEPS,
                MODEL_FOLDER_PIXEL, DATA_FOLDER_PIXEL, [-4700, 0])

    # Train DDPG agent using retina
    # agent = DdpgRetina(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION,
    #                    INIT_NOISE, FINAL_NOISE, EXPLORATION_LEN_PIXEL,
    #                    NUM_IMAGES, ACTION_DIM, REWARD_SCALE, IMAGE_SIZE)
    # agent.train(environment, INIT_EXPLORE, MAX_EPISODES_PIXEL, MAX_STEPS,
    #             MODEL_FOLDER_PIXEL, DATA_FOLDER_RETINA, [-4700, 0])
