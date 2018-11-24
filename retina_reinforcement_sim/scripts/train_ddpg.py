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
    total_episodes = 5000
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
        curr_state = trainer.state_to_tensor(env.reset())
        timestep = 0
        print "On episode : " + str(episode) + "/" + str(total_episodes)
        while not rospy.is_shutdown() and timestep < timesteps:
            timestep += 1

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

        if (episode % 1000 == 0):
            trainer.save(save_model_path, episode)

    plt.savefig(save_fig_path + "episodes_average_rewards.png")


if __name__ == '__main__':
    try:
        sys.exit(train())
    except rospy.ROSInterruptException:
        pass
