#!/usr/bin/env python

import argparse
import os
import random

import numpy as np
import torch
import rospy

from training import (BaxterPreprocessor, BaxterImagePreprocessor,
                      BaxterRetinaPreprocessor, Normalizer, Ddpg)
from model import ActorMlp, CriticMlp, ResNet6, ResNet10, WRN64, WRN128
from environment import BaxterEnvironment


def str2bool(value):
    """Convert string to boolean."""
    return value.lower() == 'true'


def lower(value):
    """Return lowercase string."""
    return value.lower()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate dynamics agent performance.')
    parser.add_argument('--network', type=lower, default=None,
                        help='[ResNet6, ResNet10, WRN64, WRN128]')
    parser.add_argument('--use-retina', type=str2bool,
                        help='if true trains using retina images')
    parser.add_argument('--epoch', type=int, default=50,
                        help='epoch of network to load')
    parser.add_argument('--episodes', type=int, default=100,
                        help='number of episodes to run')
    parser.add_argument('--steps', type=int, default=5,
                        help='number of steps in an episode')
    parser.add_argument('--visualise', type=str2bool, default=False,
                        help='if true displays images while running. No effect when not using a network.')
    args = parser.parse_args()

    rospy.init_node("evaluation process")

    # Evaluation variables
    EVAL_EP = args.episodes
    MAX_STEPS = args.steps

    # Load resnet if not using dynamics
    if args.network is not None:
        if args.network == "resnet6":
            resnet = ResNet6(2)
        elif args.network == "resnet10":
            resnet = ResNet10(2)
        elif args.network == "wrn64":
            resnet = WRN64(2)
        elif args.network == "wrn128":
            resnet = WRN128(2)
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
        state_dict = (state_dict + args.network + "/state_dicts/net_"
                      + str(args.epoch))
        resnet.load_state_dict(torch.load(state_dict))

    # Agent variables
    REPLAY_SIZE = None
    BATCH_SIZE = None
    NOISE_FUNCTION = None
    INIT_NOISE = None
    FINAL_NOISE = None
    EXPLORATION_LEN = None
    REWARD_SCALE = None
    ACTOR = ActorMlp(4, 2).cuda()
    ACTOR_OPTIM = None
    CRITIC = CriticMlp(4, 2).cuda()
    CRITIC_OPTIM = None
    if args.network is not None:
        if args.use_retina:
            PREPROCESSOR = BaxterRetinaPreprocessor(
                resnet, args.visualise, args.visualise)
        else:
            PREPROCESSOR = BaxterImagePreprocessor(
                resnet, args.visualise, args.visualise)
    else:
        PREPROCESSOR = BaxterPreprocessor(resnet)
    S_NORMALIZER = Normalizer(4)
    R_NORMALIZER = None
    MODEL_FOLDER = (os.path.dirname(os.path.realpath(__file__))
                    + "/baxter_center/mlp_normRs/state_dicts/")
    CHECKPOINT = 300000

    # Load agent from checkpoint
    agent = Ddpg(REPLAY_SIZE, BATCH_SIZE, NOISE_FUNCTION, INIT_NOISE,
                 FINAL_NOISE, EXPLORATION_LEN, REWARD_SCALE, ACTOR,
                 ACTOR_OPTIM, CRITIC, CRITIC_OPTIM, PREPROCESSOR,
                 S_NORMALIZER, R_NORMALIZER)
    agent.load_checkpoint(MODEL_FOLDER, CHECKPOINT)

    # Set seeds
    torch.manual_seed(15)
    torch.cuda.manual_seed(15)
    np.random.seed(15)
    random.seed(15)

    # Create environment
    if args.network is not None:
        env = BaxterEnvironment((1280, 800), True)
    else:
        env = BaxterEnvironment()
    rospy.on_shutdown(env.close)

    # Evaluate and print performance
    print "Performance: %0.3f +- %0.3f" % (
        agent.evaluate(env, MAX_STEPS, EVAL_EP))
