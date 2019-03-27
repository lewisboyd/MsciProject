#!/usr/bin/env python

import os
import random

import rospy
import torch
import numpy as np

from environment import BaxterEnvironment
from model import (ActorMlp, CriticMlp)
from training import (Ddpg, BaxterPreprocessor, NormalActionNoise, Normalizer)


if __name__ == '__main__':
    rospy.init_node("training process")

    # Set seeds
    torch.manual_seed(15)
    torch.cuda.manual_seed(15)
    np.random.seed(15)
    random.seed(15)

    # Environment variables
    STATE_DIM = 4
    ACTION_DIM = 2

    # Training variables
    ENVIRONMENT = BaxterEnvironment()
    INIT_EXPLORE = 2000
    MAX_STEPS = 300000
    MAX_EP_STEPS = 15
    UPDATES_PER_STEP = 1
    DATA = (os.path.dirname(os.path.realpath(__file__))
            + "/baxter_center/data/")
    DATA = None
    MODEL = (os.path.dirname(os.path.realpath(__file__))
             + "/baxter_center/mlp_normRs/state_dicts/")
    RESULT = (os.path.dirname(os.path.realpath(__file__))
              + "/baxter_center/mlp_normRs/results/")
    EVAL_FREQ = 2000
    EVAL_EP = 20
    CHECKPOINT = None

    # Agent variables
    REPLAY_SIZE = 1000000
    BATCH_SIZE = 128
    NOISE_FUNCTION = NormalActionNoise(actions=ACTION_DIM)
    INIT_NOISE = 1.0
    FINAL_NOISE = 0.02
    EXPLORATION_LEN = (MAX_STEPS * 0.75)
    REWARD_SCALE = 1.0
    ACTOR = ActorMlp(STATE_DIM, ACTION_DIM).cuda()
    ACTOR_OPTIM = torch.optim.Adam(ACTOR.parameters(), 0.0001)
    CRITIC = CriticMlp(STATE_DIM, ACTION_DIM).cuda()
    CRITIC_OPTIM = torch.optim.Adam(
        CRITIC.parameters(), 0.001, weight_decay=0.01)
    PREPROCESSOR = BaxterPreprocessor()
    S_NORMALIZER = Normalizer(STATE_DIM)
    R_NORMALIZER = None

    # Train agent
    rospy.on_shutdown(ENVIRONMENT.close)
    agent = Ddpg(REPLAY_SIZE, BATCH_SIZE, NOISE_FUNCTION, INIT_NOISE,
                 FINAL_NOISE, EXPLORATION_LEN, REWARD_SCALE, ACTOR,
                 ACTOR_OPTIM, CRITIC, CRITIC_OPTIM, PREPROCESSOR,
                 S_NORMALIZER, R_NORMALIZER)
    agent.train(ENVIRONMENT, INIT_EXPLORE, MAX_STEPS, MAX_EP_STEPS,
                UPDATES_PER_STEP, MODEL, RESULT, DATA, EVAL_FREQ,
                EVAL_EP, CHECKPOINT)
