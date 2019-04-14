#!/usr/bin/env python

import os
import random

import rospy
import torch
import torch.nn as nn
import numpy as np

from environment import BaxterEnvironment
from model import ActorMlp, CriticMlp, ResNet6, ResNet10, WRN64, WRN128
from training import (Ddpg, BaxterImagePreprocessor, BaxterRetinaPreprocessor,
                      NormalActionNoise, Normalizer)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


def str2bool(value):
    """Convert string to boolean."""
    return value.lower() == 'true'


def lower(value):
    """Returns lowercase string"""
    return value.lower()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train State Representation Net.')
    parser.add_argument('--network', type=lower, required=True,
                        help='[ResNet6, ResNet10, WRN64, WRN128]')
    parser.add_argument('--use-retina', type=str2bool, required=True,
                        help='if true trains using retina images')
    parser.add_argument('--epoch', type=int, default=50,
                        help='epoch of network to load')
    args = parser.parse_args()

    rospy.init_node("training process")

    # Set seeds
    torch.manual_seed(15)
    torch.cuda.manual_seed(15)
    np.random.seed(15)
    random.seed(15)

    # Instansiate resnet in eval mode
    if args.network == "resnet6":
        resnet = ResNet6(2)
        features = 128
    elif args.network == "resnet10":
        resnet = ResNet10(2)
        features = 512
    elif args.network == "wrn64":
        resnet = WRN64(2)
        features = 64
    elif args.network == "wrn128":
        resnet = WRN128(2)
        features = 128
    else:
        print "%s is not a valid network, choices are " % (
            args.network, "[ResNet6, ResNet10, WRN64, WRN128]")
        exit()
    resnet = resnet.cuda().eval()

    # Load state dictionary
    state_dict = (os.path.dirname(os.path.realpath(__file__))
                  + "/baxter_center/")
    if args.use_retina:
        state_dict = state_dict + "retina_"
    state_dict = (state_dict + args.network + "/state_dict/" + "net_"
                  + str(args.epoch))
    resnet.load_state_dict(torch.load(state_dict))

    # Remove final layer
    layers = list(resnet.children())[:-1]
    layers.append(Flatten())
    resnet = nn.Sequential(*layers)

    # Environment variables
    STATE_DIM = features + 2
    ACTION_DIM = 2

    # Training variables
    ENVIRONMENT = BaxterEnvironment((1280, 800), True)
    INIT_EXPLORE = 10000
    MAX_STEPS = 300000
    MAX_EP_STEPS = 15
    UPDATES_PER_STEP = 1
    DATA = (os.path.dirname(os.path.realpath(__file__))
            + "/baxter_center/data/")
    DATA = None
    MODEL = (os.path.dirname(os.path.realpath(__file__))
             + "/baxter_center/mlp_")
    RESULT = (os.path.dirname(os.path.realpath(__file__))
              + "/baxter_center/mlp_")
    if args.use_retina:
        MODEL = MODEL + "retina_"
        RESULT = RESULT + "retina_"
    MODEL = MODEL + str(features) + "vector/state_dicts/"
    RESULT = RESULT + str(features) + "vector/results/"
    EVAL_FREQ = 2000
    EVAL_EP = 20
    CHECKPOINT = None

    # Agent variables
    REPLAY_SIZE = 150000
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
    if args.use_retina:
        PREPROCESSOR = BaxterRetinaPreprocessor(resnet)
    else:
        PREPROCESSOR = BaxterImagePreprocessor(resnet)
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
