#!/usr/bin/env python

import os
from environment import PendulumPixel, PendulumLow
from training import DDPGLow, DDPGPixel, \
    OrnsteinUhlenbeckActionNoise, train


if __name__ == '__main__':
    # Training variables
    INIT_EXPLORE = 500
    MAX_EPISODES_PIXEL = 5000
    MAX_EPISODES_LOW = 400
    MAX_STEPS = 200
    EVAL_FREQ_LOW = 10

    # Agent variables
    REPLAY_SIZE = 100000
    BATCH_SIZE = 1028
    NOISE_FUNCTION = OrnsteinUhlenbeckActionNoise(1)
    INIT_NOISE = 1
    FINAL_NOISE = 0.02
    EXPLORATION_LEN_PIXEL = 100000
    EXPLORATION_LEN_LOW = 10000
    STATE_DIM_LOW = 3
    ACTION_DIM = 1

    # Save paths
    MODEL_FOLDER_NORMAL = (os.path.dirname(
        os.path.realpath(__file__)) + "/state_dicts/pendulum_normal/")
    MODEL_FOLDER_PIXEL = (os.path.dirname(
        os.path.realpath(__file__)) + "/state_dicts/pendulum_pixel/")
    MODEL_FOLDER_RETINA = (os.path.dirname(
        os.path.realpath(__file__)) + "/state_dicts/pendulum_retina/")
    DATA_FOLDER_NORMAL = (os.path.dirname(os.path.realpath(__file__))
                          + "/results/pendulum_normal/")
    DATA_FOLDER_PIXEL = (os.path.dirname(os.path.realpath(__file__))
                         + "/results/pendulum_pixel/")
    DATA_FOLDER_RETINA = (os.path.dirname(os.path.realpath(__file__))
                          + "/results/pendulum_retina/")

    # Train DDPG agent on pendulum low state dimension
    # agent = DDPGLow(REPLAY_SIZE, BATCH_SIZE, NOISE_FUNCTION, INIT_NOISE,
    #                 FINAL_NOISE, EXPLORATION_LEN_LOW, STATE_DIM_LOW,
    #                 ACTION_DIM)
    # environment = PendulumLow()
    # train(environment, agent, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
    #             MODEL_FOLDER_NORMAL, DATA_FOLDER_NORMAL,
    #             eval_freq=EVAL_FREQ_LOW)

    # Train DDPG agent on pendulum without retina
    agent = DDPGPixel(REPLAY_SIZE, BATCH_SIZE, NOISE_FUNCTION,
                      INIT_NOISE, FINAL_NOISE, EXPLORATION_LEN_PIXEL, 3, 1)
    environment = PendulumPixel(False)
    train(environment, agent, INIT_EXPLORE, MAX_EPISODES_PIXEL, MAX_STEPS,
          MODEL_FOLDER_PIXEL, DATA_FOLDER_PIXEL)

    # Train DDPG agent on pendulum with retina
    # agent = DDPGPixel(REPLAY_SIZE, BATCH_SIZE, NOISE_FUNCTION,
    #                   INIT_NOISE, FINAL_NOISE, EXPLORATION_LEN_PIXEL, 3, 1)
    # environment = PendulumPixel(True)
    # train(environment, agent, INIT_EXPLORE, MAX_EPISODES_PIXEL, MAX_STEPS,
    #             MODEL_FOLDER_PIXEL, DATA_FOLDER_RETINA)
