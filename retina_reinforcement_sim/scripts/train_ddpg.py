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
    agent = DDPG(batch_size=32)
    save_model_path = (os.path.dirname(
        os.path.realpath(__file__)) + "/state_dicts/")
    save_fig_path = (os.path.dirname(os.path.realpath(__file__))
                     + "/results/")
    total_episodes = 1000
    timesteps = 10
    avg_rewards = []

    rospy.on_shutdown(env.shutdown)

    """Train using DDPG algorithm."""
    plt.figure(0)
    plt.show(block=False)
    episode = 0
    while not rospy.is_shutdown() and episode < total_episodes:
        episode += 1
        reward_sum = 0.0
        curr_state = env.reset()
        timestep = 0
        print "On episode : " + str(episode) + "/" + str(total_episodes)
        while not rospy.is_shutdown() and timestep < timesteps:
            timestep += 1

            action_values = agent.get_exploration_action(curr_state)
            next_state, reward, terminal_state = (
                env.step(*action_values))
            print "Took action : {}".format(action_values)
            print "Got reward : {}".format(reward)
            agent.memory.push(curr_state, action_values, next_state, [reward],
                              [terminal_state])
            agent.optimise()

            curr_state = next_state
            reward_sum += reward

            if terminal_state:
                break

        reward_avg = reward_sum / timestep
        print "Avg. Reward : {}, Total Reward : {}, Steps : {}".format(
            reward_avg, reward_sum, timestep)
        avg_rewards.append(reward_avg)

        plt.clf()
        plt.title('Training')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
        plt.draw()
        plt.pause(0.01)

        if (episode % 100 == 0):
            agent.save(save_model_path, episode)

    plt.savefig(save_fig_path + "episodes_average_rewards.png")


if __name__ == '__main__':
    try:
        sys.exit(train())
    except rospy.ROSInterruptException:
        pass
