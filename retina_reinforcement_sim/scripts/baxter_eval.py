#!/usr/bin/env python

import os
import random

import numpy as np
import torch
import rospy

from training import (BaxterPreprocessor, BaxterImagePreprocessor,
                      BaxterRetinaPreprocessor, Normalizer, Ddpg)
from model import ActorMlp, CriticMlp, ResNet10
from environment import BaxterEnvironment


def set_seeds():
    """Set seeds for all libraries."""
    torch.manual_seed(15)
    torch.cuda.manual_seed(15)
    np.random.seed(15)
    random.seed(15)


def eval_mlp(agent, eval_ep, max_steps):
    """Evalutes performance using dynamics."""
    # Initialise environment
    set_seeds()
    env = BaxterEnvironment()
    rospy.on_shutdown(env.close)

    # Set preprocessor then evaluate
    agent.preprocessor = BaxterPreprocessor()
    return agent.evaluate(env, max_steps, eval_ep)


def eval_img(agent, eval_ep, max_steps):
    """Evalutes performance using ResNet on normal images to locate block."""
    # Initialise environment
    set_seeds()
    env = BaxterEnvironment((1280, 800), True)
    rospy.on_shutdown(env.close)

    # Set preprocessor then evaluate
    srnet = ResNet10(2).cuda().eval()
    state_dict = (os.path.dirname(os.path.realpath(__file__))
                  + "/baxter_center/sr/state_dicts/net_50")
    srnet.load_state_dict(torch.load(state_dict))
    agent.preprocessor = BaxterImagePreprocessor(srnet)
    return agent.evaluate(env, max_steps, eval_ep)


def eval_retina(agent, eval_ep, max_steps):
    """Evalutes performance using ResNet on retina images to locate block."""
    # Initialise environment
    set_seeds()
    env = BaxterEnvironment((1280, 800), True)
    rospy.on_shutdown(env.close)

    # Set preprocessor then evaluate
    srnet = ResNet10(2).cuda().eval()
    state_dict = (os.path.dirname(os.path.realpath(__file__))
                  + "/baxter_center/sr_retina/state_dicts/net_50")
    srnet.load_state_dict(torch.load(state_dict))
    agent.preprocessor = BaxterRetinaPreprocessor(srnet)
    return agent.evaluate(env, max_steps, eval_ep)


if __name__ == '__main__':
    rospy.init_node("evaluation process")

    # Evaluation variables
    EVAL_EP = 500
    MAX_STEPS = 5

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
    PREPROCESSOR = None
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

    # Evaluate performance of each setup
    print "Mlp Performance: %0.3f +- %0.3f" % (
        eval_mlp(agent, EVAL_EP, MAX_STEPS))
    print "Img Performance: %0.3f +- %0.3f" % (
        eval_img(agent, EVAL_EP, MAX_STEPS))
    print "Retina Performance: %0.3f +- %0.3f" % (
        eval_retina(agent, EVAL_EP, MAX_STEPS))
