from environment import PendulumEnvironment
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import random
from training import NormalActionNoise
import matplotlib.pyplot as plt


def update_plot(plot, timestep, ep_reward):
    plot.set_xdata(np.append(plot.get_xdata(), timestep))
    plot.set_ydata(np.append(plot.get_ydata(), ep_reward))
    plt.draw()
    plt.pause(0.01)


def draw_plot(total_timesteps):
    plt.ion()
    plt.show(block=False)

    ax = plt.subplot(311)
    ax.set_title('Evaluation Performance')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Episode Reward')
    ax.set_xlim(0, total_timesteps)
    ax.set_ylim(-17, 0)
    reward_plot, = ax.plot(0, 0)

    return reward_plot


MAX_EPISODES = 1000
MAX_STEPS = 200
REPLAY_SIZE = 1000
BATCH_SIZE = 16
NOISE_FUNCTION = NormalActionNoise(0, 0.2, 1)
INIT_NOISE = 1
FINAL_NOISE = 0.02
EXPLORATION_LEN = 10000

# Initialise agent and environment
trainer = DDPG(REPLAY_SIZE, BATCH_SIZE, NOISE_FUNCTION, INIT_NOISE, FINAL_NOISE,
               EXPLORATION_LEN)
env = PendulumEnvironment()
reward_plot = draw_plot(MAX_EPISODES * MAX_STEPS)

for ep in range(MAX_EPISODES):
    # Reset environment and get intial state
    state = env.reset()
    state = trainer.state_to_tensor(state)
    ep_reward = 0

    print "EPISODE : ", ep
    for step in range(MAX_STEPS):
        # If validation don't use noise, save experiences or optimise
        if ep % 10 == 0:
            action = trainer.get_exploitation_action(state)
            new_state, reward = env.step(action)
            ep_reward += reward
            new_state = trainer.state_to_tensor(new_state)
            done = step == MAX_STEPS - 1
            # When done plot performance
            if done:
                update_plot(reward_plot, ep * MAX_STEPS, ep_reward)
        else:
            # Step using noise action
            action = trainer.get_exploration_action(state)
            new_state, reward = env.step(action)
            ep_reward += reward
            done = step == MAX_STEPS - 1

            # Convert to tensors and save
            new_state = trainer.state_to_tensor(new_state)
            reward = trainer.reward_to_tensor(reward)
            done = trainer.done_to_tensor(done)
            trainer.memory.push(state, action, new_state, reward, done)

            # Optimise models
            trainer.optimise()

            # Update current state
            state = new_state

    print "Total Reward : ", ep_reward
