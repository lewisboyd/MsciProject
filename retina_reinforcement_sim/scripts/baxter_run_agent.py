#!/usr/bin/env python

import os
import time

import torch
import rospy

from training import (BaxterPreprocessor, NormalActionNoise)
from model import ActorMlp, FeatureExtractor
from environment import BaxterEnvironment


if __name__ == '__main__':
    STATE_DIM = 4
    ACTION_DIM = 2
    ACTOR_WEIGHTS = (os.path.dirname(os.path.realpath(__file__))
                     + "/baxter_center/mlp_normRs/state_dicts/300000_actor")
    SR_WEIGHTS = (os.path.dirname(os.path.realpath(__file__))
                  + "/baxter_center/sr/100_net")

    srnet = FeatureExtractor(1, 2).cuda()
    srnet.load_state_dict(torch.load(SR_WEIGHTS))
    actor = ActorMlp(STATE_DIM, ACTION_DIM).cuda()
    actor.load_state_dict(torch.load(ACTOR_WEIGHTS))

    # Create preprocessors
    preprocessor = BaxterPreprocessor()

    # Create environment
    rospy.init_node('data_gathering_process')
    env = BaxterEnvironment()
    rospy.on_shutdown(env.close)

    arm_pos = torch.tensor([env.x_pos, env.y_pos])
    image = torch.tensor(env.image).cuda().permute(2, 0, 1)
    object_dist = srnet(image)
    state = torch.cat((object_dist, arm_pos), 1)
    timestep_ep = 0
    done = False
    while True:

        if done:
            arm_pos = torch.tensor([env.x_pos, env.y_pos])
            image = torch.tensor(env.image).cuda().permute(2, 0, 1)
            object_dist = srnet(image)
            state = torch.cat((object_dist, arm_pos), 1)
            timestep_ep = 0
            done = False

        action = actor(state)

    # Populate tensors
    index = 0
    timestep_ep = 0
    done = False
    state = preprocessor(env.reset()).to(device)
    image = env.image
    while index < SIZE:
        # If episode finished start and new one and reset noise function
        if done:
            state = preprocessor(env.reset()).to(device)
            image = env.image
            noise_function.reset()
            timestep_ep = 0

        # Step through environment using noise function
        next_obs, _, done = env.step(np.clip(noise_function(), -1, 1))
        done = done or (timestep_ep == MAX_EP_STEPS)
        next_state = preprocessor(next_obs).to(device)

        # Add data to tensors
        states[index] = state
        images[index] = torch.from_numpy(image).permute(2, 0, 1).to(device)

        # Advance state and indices
        state = next_state
        timestep_ep = timestep_ep + 1
        index = index + 1

    # Save data
    torch.save(states, DATA_FOLDER + "states")
    torch.save(images, DATA_FOLDER + "images")

    # Close environment
    env.close()

    end = time.time()
    mins = (end - start) / 60
    print "Gathered %d state-image pairs in %d hours %d minutes" % (
        SIZE, mins / 60, mins % 60)
