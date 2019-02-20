#!/usr/bin/env python

import os

import torch

from environment import PendulumPixel, PendulumLow
from model import FeatureExtractor
from training import (DdpgMlp, DdpgCnn, DdpgSr, OrnsteinUhlenbeckActionNoise,
                      ImagePreprocessor, RetinaPreprocessor)


if __name__ == '__main__':
    # Training variables
    INIT_EXPLORE = 0
    MAX_STEPS = 200
    DATA_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/data/")

    # Training variables low state
    MAX_EPISODES_LOW = 400
    EVAL_FREQ_LOW = 20

    # Training variables image
    MAX_EPISODES_PIXEL = 2000

    # Agent variables
    REPLAY_SIZE = 100000
    REWARD_SCALE = 0.1
    ACTION_DIM = 1
    NOISE_FUNCTION = OrnsteinUhlenbeckActionNoise(ACTION_DIM)
    INIT_NOISE = 1
    FINAL_NOISE = 0.02

    # Agent variables low state
    BATCH_SIZE_LOW = 64
    STATE_DIM_LOW = 3
    EXPLORATION_LEN_LOW = 10000

    # Agent variables image
    BATCH_SIZE_IMAGE = 16
    NUM_IMAGES = 3
    IMAGE_SIZE = (500, 500)
    EXPLORATION_LEN_PIXEL = 100000

    # Save paths for state_dicts
    MODEL_FOLDER_NORMAL = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/state_dicts/normal/")
    MODEL_FOLDER_PIXEL = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/state_dicts/pixel/")
    MODEL_FOLDER_RETINA = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/state_dicts/retina/")
    MODEL_FOLDER_PIXEL_SR_FROZEN = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/state_dicts/pixel_sr_frozen/")
    MODEL_FOLDER_PIXEL_SR = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/state_dicts/pixel_sr/")

    # Save paths for performance data
    RESULT_FOLDER_NORMAL = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/results/low_dim/")
    RESULT_FOLDER_PIXEL = (os.path.dirname(os.path.realpath(__file__))
                           + "/pendulum/results/pixel/")
    RESULT_FOLDER_RETINA = (os.path.dirname(os.path.realpath(__file__))
                            + "/pendulum/results/retina/")
    RESULT_FOLDER_PIXEL_SR_FROZEN = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/results/pixel_sr_frozen/")
    RESULT_FOLDER_PIXEL_SR = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/results/pixel_sr/")

    # Path to state_dict for state represenation net
    SR_STATE_DICT = (os.path.dirname(os.path.realpath(__file__))
                     + "/pendulum/state_dicts/pixel_supervised/net_20")

    # Path to folder containing data for experience replay
    DATA_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/data/")

    # Create Pendulum environment returning 3 images
    environment = PendulumPixel()

    # Train DDPG agent using CNN
    # agent = DdpgCnn(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION,
    #                 INIT_NOISE, FINAL_NOISE, EXPLORATION_LEN_PIXEL,
    #                 REWARD_SCALE, NUM_IMAGES, ACTION_DIM)
    # agent.train(environment, INIT_EXPLORE, MAX_EPISODES_PIXEL, MAX_STEPS,
    #             MODEL_FOLDER_PIXEL, RESULT_FOLDER_PIXEL, [-5000, 0])

    # Train DDPG agent using retina
    # preprocessor = RetinaPreprocessor(*image_size)
    # agent = DdpgCnn(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION,
    #                 INIT_NOISE, FINAL_NOISE, EXPLORATION_LEN_PIXEL,
    #                 REWARD_SCALE, NUM_IMAGES, ACTION_DIM, preprocessor)
    # agent.train(environment, INIT_EXPLORE, MAX_EPISODES_PIXEL, MAX_STEPS,
    #             MODEL_FOLDER_PIXEL, RESULT_FOLDER_RETINA, [-5000, 0])

    # Train DDPG agent using state represenation net with parameters frozen
    preprocessor = ImagePreprocessor(NUM_IMAGES)
    sr_net = FeatureExtractor(NUM_IMAGES, STATE_DIM_LOW)
    sr_net.load_state_dict(torch.load(SR_STATE_DICT))
    for param in sr_net.parameters():
        param.requires_grad = False
    agent = DdpgSr(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
                   FINAL_NOISE, EXPLORATION_LEN_PIXEL, REWARD_SCALE, sr_net,
                   STATE_DIM_LOW, ACTION_DIM, preprocessor)
    agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
                MODEL_FOLDER_PIXEL_SR, RESULT_FOLDER_PIXEL_SR,
                data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
                eval_freq=EVAL_FREQ_LOW)

    # Train DDPG agent using state representation net with parameters trainable
    sr_net.load_state_dict(torch.load(SR_STATE_DICT))
    for param in sr_net.parameters():
        param.requires_grad = True
    agent = DdpgSr(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
                   FINAL_NOISE, EXPLORATION_LEN_PIXEL, REWARD_SCALE, sr_net,
                   STATE_DIM_LOW, ACTION_DIM, preprocessor)
    agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
                MODEL_FOLDER_PIXEL_SR_FROZEN, RESULT_FOLDER_PIXEL_SR_FROZEN,
                data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
                eval_freq=EVAL_FREQ_LOW)

    # Train DDPG agent on pendulum low state dimension
    agent = DdpgMlp(REPLAY_SIZE, BATCH_SIZE_LOW, NOISE_FUNCTION, INIT_NOISE,
                    FINAL_NOISE, EXPLORATION_LEN_LOW, REWARD_SCALE,
                    STATE_DIM_LOW, ACTION_DIM)
    environment = PendulumLow()
    agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
                MODEL_FOLDER_NORMAL, RESULT_FOLDER_NORMAL,
                data_folder=DATA_FOLDER, plot_ylim=[-2000, 0],
                eval_freq=EVAL_FREQ_LOW)
