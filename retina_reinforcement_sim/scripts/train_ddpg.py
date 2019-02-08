#!/usr/bin/env python

from environment import Environment
from training import DDPG
import sys
import os
import rospy
import matplotlib.pyplot as plt
import numpy as np


def update_plot(plot, x_data, y_data):
    plot.set_xdata(np.append(plot.get_xdata(), x_data))
    plot.set_ydata(np.append(plot.get_ydata(), y_data))
    plt.draw()
    plt.pause(0.01)


def draw_plots(total_timesteps):
    plt.ion()
    plt.show(block=False)

    ax = plt.subplot(311)
    ax.set_title('Episode Reward')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Average Reward')
    ax.set_xlim(0, total_timesteps)
    ax.set_ylim(-1, 1)
    reward_plot, = ax.plot(0, 0)

    ax = plt.subplot(312)
    ax.set_title('Critic Loss')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, total_timesteps)
    ax.set_ylim(0, 10)
    critic_plot, = ax.plot(0, 0)

    ax = plt.subplot(313)
    ax.set_title('Actor Loss')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, total_timesteps)
    ax.set_ylim(-10, 10)
    actor_plot, = ax.plot(0, 0)

    return reward_plot, critic_plot, actor_plot


def train():
    """Initialise node, environment and agent then starts training."""
    rospy.init_node("trainer")

    total_timesteps = 40000
    max_episode_len = 10
    save_model_path = (os.path.dirname(
        os.path.realpath(__file__)) + "/state_dicts/")
    save_fig_path = (os.path.dirname(os.path.realpath(__file__))
                     + "/results/")

    env = Environment()
    trainer = DDPG(memory_capacity=3500, batch_size=200, exploration_len=20000)

    rospy.on_shutdown(env.shutdown)

    """Train using DDPG algorithm."""
    reward_plot, critic_plot, actor_plot = draw_plots(total_timesteps)
    timestep = 0
    while not rospy.is_shutdown() and timestep < total_timesteps:
        episode_timestep = 0
        reward_sum = 0.0
        curr_state = trainer.state_to_tensor(env.reset())
        while not rospy.is_shutdown() and episode_timestep < max_episode_len:
            timestep += 1
            episode_timestep += 1

            action = trainer.get_exploration_action(curr_state)
            next_state, reward, done = env.step(*action[0])
            reward_sum += reward

            next_state, reward, done = trainer.interpret(
                next_state, reward, done)
            trainer.memory.push(curr_state, action, next_state,
                                reward, done)

            critic_loss, actor_loss = trainer.optimise()
            if critic_loss is not None:
                update_plot(critic_plot, timestep, critic_loss)
                update_plot(actor_plot, timestep, actor_loss)

            curr_state = next_state
            if not done:    # Because the trainer converts True to 0
                break

        update_plot(reward_plot, timestep, reward_sum / episode_timestep)

        if (timestep % 4000 == 0):
            trainer.save(save_model_path, timestep)

    plt.savefig(save_fig_path + "training_results.png")


if __name__ == '__main__':
    try:
        sys.exit(train())
    except rospy.ROSInterruptException:
        pass
