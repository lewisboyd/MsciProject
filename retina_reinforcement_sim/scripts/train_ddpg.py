#!/usr/bin/env python

from environment import Environment
from training import DDPG
import sys
import os
import rospy
import matplotlib.pyplot as plt


def train():
    """Initialise node, environment and agent then starts training."""
    rospy.init_node("trainer")

    env = Environment()
    trainer = DDPG()
    save_model_path = (os.path.dirname(
        os.path.realpath(__file__)) + "/state_dicts/")
    save_fig_path = (os.path.dirname(os.path.realpath(__file__))
                     + "/results/")
    total_timesteps = 40000
    max_episode_len = 10
    reward_timestep = [0]
    avg_rewards = [-1]

    rospy.on_shutdown(env.shutdown)

    """Train using DDPG algorithm."""
    plt.figure(0)
    plt.show(block=False)
    timestep = 0
    while not rospy.is_shutdown() and timestep < total_timesteps:
        timestep += 1
        reward_sum = 0.0
        curr_state = trainer.state_to_tensor(env.reset())
        episode_timestep = 0
        while not rospy.is_shutdown() and episode_timestep < max_episode_len:
            episode_timestep += 1

            action = trainer.get_exploration_action(curr_state)
            next_state, reward, done = env.step(*action[0])
            reward_sum += reward

            next_state, reward, done = trainer.interpret(
                next_state, reward, done)
            trainer.memory.push(curr_state, action, next_state,
                                reward, done)
            trainer.optimise()

            curr_state = next_state
            if not done:    # Because the trainer converts True to 0
                break

        reward_avg = reward_sum / episode_timestep
        reward_timestep.append(timestep)
        avg_rewards.append(reward_avg)

        plt.clf()
        plt.title('Training')
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        plt.plot(reward_timestep, avg_rewards)
        plt.axis([0, total_timesteps, -1, 1])
        plt.draw()
        plt.pause(0.01)

        if (timestep % 4000 == 0):
            trainer.save(save_model_path, timestep)

    plt.savefig(save_fig_path + "episodes_average_rewards.png")


if __name__ == '__main__':
    try:
        sys.exit(train())
    except rospy.ROSInterruptException:
        pass
