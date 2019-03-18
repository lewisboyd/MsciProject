#!/usr/bin/env python

import os
import copy

import torch

from environment import PendulumPixel, PendulumLow
from model import (ActorCnn, CriticCnn, ActorSr, CriticSr,
                   ActorMlp, CriticMlp, FeatureExtractor)
from training import (Ddpg, OrnsteinUhlenbeckActionNoise,
                      PendulumPreprocessor, ImagePreprocessor)


if __name__ == '__main__':
    # Training variables
    INIT_EXPLORE = 0
    MAX_STEPS = 200
    MAX_STEPS = 200
    EVAL_EP = 10
    DATA_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/data/")

    # Training variables low state
    MAX_EPISODES_LOW = 300
    EVAL_FREQ_LOW = 5

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
    MODEL_MLP = (os.path.dirname(os.path.realpath(__file__))
                 + "/pendulum/mlp_normRs_multistep/state_dicts/")
    MODEL_CNN_PRE2 = (os.path.dirname(os.path.realpath(__file__))
                      + "/pendulum/cnn_pre2Frozen_lowRs/state_dicts/")
    MODEL_RETINA = (os.path.dirname(os.path.realpath(__file__))
                    + "/pendulum/retina/state_dicts/")
    MODEL_CNN_SR1_FROZEN = (os.path.dirname(os.path.realpath(__file__))
                            + "/pendulum/cnn_sr1Frozen_lowRs/state_dicts/")
    MODEL_CNN_SR1 = (os.path.dirname(os.path.realpath(__file__))
                     + "/pendulum/cnn_sr1_lowRs/state_dicts/")
    MODEL_CNN_SR2_FROZEN = (os.path.dirname(os.path.realpath(__file__))
                            + "/pendulum/cnn_sr2Frozen_lowRs/state_dicts/")
    MODEL_CNN_SR2 = (os.path.dirname(os.path.realpath(__file__))
                     + "/pendulum/cnn_sr2_lowRs/state_dicts/")

    # Save paths for performance data
    RESULT_MLP = (os.path.dirname(os.path.realpath(__file__))
                  + "/pendulum/mlp_normRs_multistep/results/")
    RESULT_CNN_PRE2 = (os.path.dirname(os.path.realpath(__file__))
                       + "/pendulum/cnn_pre2Frozen_lowRs/results/")
    RESULT_RETINA = (os.path.dirname(os.path.realpath(__file__))
                     + "/pendulum/retina/results/")
    RESULT_CNN_SR1_FROZEN = (os.path.dirname(os.path.realpath(__file__))
                             + "/pendulum/cnn_sr1Frozen_lowRs/results/")
    RESULT_CNN_SR1 = (os.path.dirname(os.path.realpath(__file__))
                      + "/pendulum/cnn_sr1_lowRs/results/")
    RESULT_CNN_SR2_FROZEN = (os.path.dirname(os.path.realpath(__file__))
                             + "/pendulum/cnn_sr2Frozen_lowRs/results/")
    RESULT_CNN_SR2 = (os.path.dirname(os.path.realpath(__file__))
                      + "/pendulum/cnn_sr2_lowRs/results/")

    # Path to state_dict for state represenation net
    SR1_STATE_DICT = (os.path.dirname(os.path.realpath(__file__))
                      + "/pendulum/sr1/state_dicts/net_90")
    SR2_STATE_DICT = (os.path.dirname(os.path.realpath(__file__))
                      + "/pendulum/sr2/state_dicts/net_100")

    # Create low state pendulum environment and its preprocessor
    environment = PendulumLow()
    preprocessor = PendulumPreprocessor()

    # DDPG agent using low state dimension
    actor = ActorMlp(STATE_DIM_LOW, ACTION_DIM).cuda()
    actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)
    critic = CriticMlp(STATE_DIM_LOW, ACTION_DIM).cuda()
    critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
                                    weight_decay=0.01)
    # agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
    #              FINAL_NOISE, EXPLORATION_LEN_IMAGE, 1, actor,
    #              actor_optim, critic, critic_optim, preprocessor)
    # agent.train(environment, 500, MAX_EPISODES_LOW, MAX_STEPS,
    #             MODEL_MLP, RESULT_MLP,
    #             data_folder=None, plot_ylim=[-2000, 0],
    #             eval_freq=EVAL_FREQ_LOW, eval_ep=EVAL_EP)
    agent = Ddpg(REPLAY_SIZE, 64, NOISE_FUNCTION, INIT_NOISE,
                 FINAL_NOISE, 10000, 1, actor,
                 actor_optim, critic, critic_optim, preprocessor)
    agent.train(environment, 0, 20000, 200, 40,
                MODEL_MLP, RESULT_MLP,
                data_folder=DATA_FOLDER, plot_ylim=[-2000, 0],
                eval_freq=10, eval_ep=5)

    # Create image pendulum environment and its preprocessor
    # environment = PendulumPixel()
    # preprocessor = ImagePreprocessor(NUM_IMAGES)

    # DDPG agent using state represenation 1 net with parameters frozen
    # sr_net = FeatureExtractor(NUM_IMAGES, STATE_DIM_LOW)
    # sr_net.load_state_dict(torch.load(SR1_STATE_DICT))
    # for param in sr_net.parameters():
    #     param.requires_grad = False
    # actor = ActorSr(sr_net, STATE_DIM_LOW, ACTION_DIM).cuda()
    # actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)
    # critic = CriticSr(copy.deepcopy(sr_net), STATE_DIM_LOW,
    #                   ACTION_DIM).cuda()
    # critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
    #                                 weight_decay=0.01)
    # agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
    #              FINAL_NOISE, EXPLORATION_LEN_IMAGE, REWARD_SCALE, actor,
    #              actor_optim, critic, critic_optim, preprocessor)
    # agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
    #             MODEL_CNN_SR1_FROZEN, RESULT_CNN_SR1_FROZEN,
    #             data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
    #             eval_freq=EVAL_FREQ_LOW, eval_ep=EVAL_EP)

    # DDPG agent using state representation 1 net with parameters trainable
    # sr_net = FeatureExtractor(NUM_IMAGES, STATE_DIM_LOW)
    # sr_net.load_state_dict(torch.load(SR1_STATE_DICT))
    # for param in sr_net.parameters():
    #     param.requires_grad = True
    # actor = ActorSr(sr_net, STATE_DIM_LOW, ACTION_DIM).cuda()
    # actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)
    # critic = CriticSr(copy.deepcopy(sr_net), STATE_DIM_LOW,
    #                   ACTION_DIM).cuda()
    # critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
    #                                 weight_decay=0.01)
    # agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
    #              FINAL_NOISE, EXPLORATION_LEN_IMAGE, REWARD_SCALE, actor,
    #              actor_optim, critic, critic_optim, preprocessor)
    # agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
    #             MODEL_CNN_SR1, RESULT_CNN_SR1,
    #             data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
    #             eval_freq=EVAL_FREQ_LOW, eval_ep=EVAL_EP)

    # DDPG agent using state represenation 2 net with parameters frozen
    # sr_net = FeatureExtractor2(NUM_IMAGES, STATE_DIM_LOW)
    # sr_net.load_state_dict(torch.load(SR2_STATE_DICT))
    # for param in sr_net.parameters():
    #     param.requires_grad = False
    # actor = ActorSr(sr_net, STATE_DIM_LOW, ACTION_DIM).cuda()
    # actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)
    # critic = CriticSr(copy.deepcopy(sr_net), STATE_DIM_LOW,
    #                   ACTION_DIM).cuda()
    # critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
    #                                 weight_decay=0.01)
    # agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
    #              FINAL_NOISE, EXPLORATION_LEN_IMAGE, REWARD_SCALE, actor,
    #              actor_optim, critic, critic_optim, preprocessor)
    # agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
    #             MODEL_CNN_SR2_FROZEN, RESULT_CNN_SR2_FROZEN,
    #             data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
    #             eval_freq=EVAL_FREQ_LOW, eval_ep=EVAL_EP)

    # DDPG agent using state representation 2 net with parameters trainable
    # sr_net = FeatureExtractor2(NUM_IMAGES, STATE_DIM_LOW)
    # sr_net.load_state_dict(torch.load(SR2_STATE_DICT))
    # for param in sr_net.parameters():
    #     param.requires_grad = True
    # actor = ActorSr(sr_net, STATE_DIM_LOW, ACTION_DIM).cuda()
    # actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)
    # critic = CriticSr(copy.deepcopy(sr_net), STATE_DIM_LOW,
    #                   ACTION_DIM).cuda()
    # critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
    #                                 weight_decay=0.01)
    # agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
    #              FINAL_NOISE, EXPLORATION_LEN_IMAGE, REWARD_SCALE, actor,
    #              actor_optim, critic, critic_optim, preprocessor)
    # agent.train(environment, INIT_EXPLORE, MAX_EPISODES_LOW, MAX_STEPS,
    #             MODEL_CNN_SR2, RESULT_CNN_SR2,
    #             data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
    #             eval_freq=EVAL_FREQ_LOW, eval_ep=EVAL_EP)

    # DDPG agent using frozen pretrained convolutional layers from sr2
    # actor = ActorCnn(NUM_IMAGES, ACTION_DIM).cuda()
    # critic = CriticCnn(NUM_IMAGES, ACTION_DIM).cuda()
    # state_dict = torch.load(SR2_STATE_DICT)
    # for key, value in state_dict.items():
    #     if key.startswith('fc'):
    #         state_dict.pop(key, None)
    # actor.load_state_dict(state_dict, strict=False)
    # critic.load_state_dict(state_dict, strict=False)
    # del state_dict  # Release memory used
    # for name, param in actor.named_parameters():
    #     if name.startswith('conv'):
    #         param.requires_grad = False
    # for name, param in critic.named_parameters():
    #     if name.startswith('conv'):
    #         param.requires_grad = False
    # actor_optim = torch.optim.Adam(actor.parameters(), 0.0001)
    # critic_optim = torch.optim.Adam(critic.parameters(), 0.001,
    #                                 weight_decay=0.01)
    # agent = Ddpg(REPLAY_SIZE, BATCH_SIZE_IMAGE, NOISE_FUNCTION, INIT_NOISE,
    #              FINAL_NOISE, EXPLORATION_LEN_IMAGE, REWARD_SCALE, actor,
    #              actor_optim, critic, critic_optim, preprocessor)
    # agent.train(environment, INIT_EXPLORE, MAX_EPISODES_IMAGE, MAX_STEPS,
    #             MODEL_CNN_PRE2, RESULT_CNN_PRE2,
    #             data_folder=DATA_FOLDER, plot_ylim=[-5000, 0],
    #             eval_freq=EVAL_FREQ_IMAGE, eval_ep=EVAL_EP)
