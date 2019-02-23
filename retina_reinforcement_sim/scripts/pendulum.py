#!/usr/bin/env python

import os
import copy

import torch

from environment import PendulumPixel, PendulumLow
from model import (ActorCnn, CriticCnn, ActorSr, CriticSr,
                   ActorMlp, CriticMlp, FeatureExtractor)
from training import (Ddpg, OrnsteinUhlenbeckActionNoise,
                      Preprocessor, ImagePreprocessor)


if __name__ == '__main__':
    # Training variables
    INIT_EXPLORE = 0
    MAX_STEPS = 200
    EVAL_EP = 10
    DATA_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/data/")

    # Training variables low state
    MAX_EPISODES_LOW = 100
    EVAL_FREQ_LOW = 2

    # Training variables image
    MAX_EPISODES_IMAGE = 1000
    EVAL_FREQ_IMAGE = 10

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
    EXPLORATION_LEN_IMAGE = 100000

    # Save paths for state_dicts
    MODEL_FOLDER_LOW = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/low_dim/state_dicts/")
    MODEL_FOLDER_CNN = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/cnn_pretrained/state_dicts/")
    MODEL_FOLDER_RETINA = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/retina/state_dicts/")
    MODEL_FOLDER_CNN_SR_FROZEN = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/cnn_sr_frozen/state_dicts/")
    MODEL_FOLDER_CNN_SR = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/cnn_sr/state_dicts/")

    # Save paths for performance data
    RESULT_FOLDER_LOW = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/low_dim/results/")
    RESULT_FOLDER_CNN = (os.path.dirname(os.path.realpath(__file__))
                         + "/pendulum/cnn_pretrained/results/")
    RESULT_FOLDER_RETINA = (os.path.dirname(os.path.realpath(__file__))
                            + "/pendulum/retina/results/")
    RESULT_FOLDER_CNN_SR_FROZEN = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/cnn_sr_frozen/results/")
    RESULT_FOLDER_CNN_SR = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/cnn_sr/results/")

    # Path to state_dict for state represenation net
    SR_STATE_DICT = (os.path.dirname(os.path.realpath(__file__))
                     + "/pendulum/sr/state_dicts/net_20")

    # Create low state pendulum environment and its preprocessor
    environment = PendulumLow()
    preprocessor = Preprocessor()

    # DDPG agent using low state dimension
    actor = ActorMlp(STATE_DIM_LOW, ACTION_DIM).cuda()
    actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)

    critic = CriticMlp(STATE_DIM_LOW, ACTION_DIM).cuda()
    critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
                                    weight_decay=0.01)
    agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_LOW, NOISE_FUNCTION, INIT_NOISE,
                 FINAL_NOISE, EXPLORATION_LEN_IMAGE, REWARD_SCALE, actor,
                 actor_optim, critic, critic_optim, preprocessor)
    agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
                MODEL_FOLDER_LOW, RESULT_FOLDER_LOW,
                data_folder=DATA_FOLDER, plot_ylim=[-2000, 0],
                eval_freq=EVAL_FREQ_LOW, eval_ep=EVAL_EP)

    # Create image pendulum environment and its preprocessor
    environment = PendulumPixel()
    preprocessor = ImagePreprocessor(NUM_IMAGES)

    # DDPG agent using state represenation net with parameters frozen
    sr_net = FeatureExtractor(NUM_IMAGES, STATE_DIM_LOW)
    sr_net.load_state_dict(torch.load(SR_STATE_DICT))
    for param in sr_net.parameters():
        param.requires_grad = False
    actor = ActorSr(sr_net, STATE_DIM_LOW, ACTION_DIM).cuda()
    actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)

    critic = CriticSr(copy.deepcopy(sr_net), STATE_DIM_LOW,
                      ACTION_DIM).cuda()
    critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
                                    weight_decay=0.01)
    agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
                 FINAL_NOISE, EXPLORATION_LEN_IMAGE, REWARD_SCALE, actor,
                 actor_optim, critic, critic_optim, preprocessor)
    agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
                MODEL_FOLDER_CNN_SR_FROZEN, RESULT_FOLDER_CNN_SR_FROZEN,
                data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
                eval_freq=EVAL_FREQ_LOW, eval_ep=EVAL_EP)

    # DDPG agent using state representation net with parameters trainable
    sr_net.load_state_dict(torch.load(SR_STATE_DICT))
    for param in sr_net.parameters():
        param.requires_grad = True
    actor = ActorSr(sr_net, STATE_DIM_LOW, ACTION_DIM).cuda()
    actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)

    critic = CriticSr(copy.deepcopy(sr_net), STATE_DIM_LOW,
                      ACTION_DIM).cuda()
    critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
                                    weight_decay=0.01)
    agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
                 FINAL_NOISE, EXPLORATION_LEN_IMAGE, REWARD_SCALE, actor,
                 actor_optim, critic, critic_optim, preprocessor)
    agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
                MODEL_FOLDER_CNN_SR, RESULT_FOLDER_CNN_SR,
                data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
                eval_freq=EVAL_FREQ_LOW, eval_ep=EVAL_EP)

    # DDPG agent using pretrained convolutional layers
    actor = ActorCnn(NUM_IMAGES, ACTION_DIM).cuda()
    critic = CriticCnn(NUM_IMAGES, ACTION_DIM).cuda()
    state_dict = torch.load(SR_STATE_DICT)
    state_dict.pop("fc1.weight", None)
    state_dict.pop("fc1.bias", None)
    state_dict.pop("fc2.weight", None)
    state_dict.pop("fc2.bias", None)
    state_dict.pop("fc3.weight", None)
    state_dict.pop("fc3.bias", None)
    actor.load_state_dict(state_dict, strict=False)
    critic.load_state_dict(state_dict, strict=False)
    del state_dict  # Release memory used
    actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)

    critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
                                    weight_decay=0.01)
    agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
                 FINAL_NOISE, EXPLORATION_LEN_IMAGE, REWARD_SCALE, actor,
                 actor_optim, critic, critic_optim, preprocessor)
    agent.train(environment, INIT_EXPLORE, MAX_EPISODES_IMAGE, MAX_STEPS,
                MODEL_FOLDER_CNN, RESULT_FOLDER_CNN,
                data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
                eval_freq=EVAL_FREQ_IMAGE, eval_ep=EVAL_EP)
