import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch


def train(env, agent, init_explore, max_episodes, max_steps,
          model_folder, data_folder, plot_ylim=[-200, 0], eval_freq=100,
          eval_ep=10):
    """Train the agent.

    Args:
        env (object): Environment to train agent in.
        agent (object): Agent to use for policy generation and
                       optimisation.
        init_explore (int): Number of episodes to perform random
                           exploration training.
        max_episodes (int): Maximum number of episodes to train for.
        max_steps (int): Maximum number of steps in one episode.
        model_folder (path): Folder to save models in during training.
        data_folder (path): Folder to save validation data.
        plot_ylim (array): Minimum and maximum value for episode reward axis.
        eval_freq (int): How many episodes of training before next evaluation.
        eval_ep (int): How many episodes to run for evaluation.
    """
    # Create folders for saving data
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    start = time.time()

    # Create interactive plot
    reward_plot = _create_plot(max_episodes, max_steps, plot_ylim)

    # Populate replay buffer using agents noise function
    for _ in range(init_explore):
        # Reset noise function and environment
        agent.noise_function.reset()
        state = env.reset()
        state = agent.state_to_tensor(state)

        for step in range(max_steps):
            # Step using noise value
            action = torch.tensor(agent.noise_function(),
                                  device=agent.device,
                                  dtype=torch.float).clamp(-1, 1)
            new_state, reward = env.step(action)
            done = step == max_steps - 1

            # Convert to tensors and save
            new_state = agent.state_to_tensor(new_state)
            reward = agent.reward_to_tensor(reward)
            done = agent.done_to_tensor(done)
            agent.memory.push(state, action, new_state, reward, done)

            # Update current statenoise_function
            state = new_state

    # Plot pretraining performance
    eval_reward = _evaluate(agent, env, max_steps, eval_ep)
    _update_plot(reward_plot, 0, eval_reward)
    print "Initial Performance: %f" % eval_reward

    # Train agent
    for ep in range(1, max_episodes + 1):
        # Reset noise function and environment
        agent.noise_function.reset()
        state = env.reset()
        state = agent.state_to_tensor(state)

        ep_reward = 0.
        for step in range(max_steps):
            # Step using noisey action
            action = agent.get_exploration_action(state)
            new_state, reward = env.step(action)
            done = step == max_steps - 1
            ep_reward += reward.item()

            # Convert to tensors and save
            new_state = agent.state_to_tensor(new_state)
            reward = agent.reward_to_tensor(reward)
            done = agent.done_to_tensor(done)
            agent.memory.push(state, action, new_state, reward, done)

            # Optimise agent
            agent.optimise()

            # Update current state
            state = new_state

        print "Episode: %4d/%4d  Reward: %0.2f" % (ep, max_episodes, ep_reward)

        # Evaluate performance and save agent every 100 episodes
        if ep % eval_freq == 0:
            eval_reward = _evaluate(agent, env, max_steps, eval_ep)
            _update_plot(reward_plot, ep * max_steps, eval_reward)
            agent.save(model_folder, str(ep))
            print "Evaluation Reward: %f" % eval_reward

    # Save figure and data
    plt.savefig(data_folder + "training_performance.png")
    reward_plot.get_xdata().tofile(data_folder
                                   + "eval_timesteps.txt")
    reward_plot.get_ydata().tofile(data_folder
                                   + "eval_rewards.txt")

    # Close figure and environment
    plt.clf()
    env.close()

    end = time.time()
    mins = (end - start) / 60
    print "Training finished after %d hours %d minutes" % (
        mins / 60, mins % 60)


def _evaluate(agent, env, max_steps, eval_episodes=10):
    # Average multiple episode rewards
    avg_reward = 0.
    for _ in range(eval_episodes):
        state = env.reset()
        state = agent.state_to_tensor(state)
        for step in range(max_steps):
            action = agent.get_exploitation_action(state)
            new_state, reward = env.step(action)
            avg_reward += reward
            state = agent.state_to_tensor(new_state)
    return avg_reward / eval_episodes


def _create_plot(max_episodes, max_steps, ylim):
    # Create interactive plot
    plt.ion()
    plt.show(block=False)
    ax = plt.subplot(111)
    ax.set_title('Evaluation Performance')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Episode Reward')

    # Set limits
    ax.set_xlim(0, max_episodes * max_steps)
    ax.set_ylim(*ylim)

    # Create subplot without any data
    reward_plot, = ax.plot(0, 0, 'b')
    reward_plot.set_xdata([])
    reward_plot.set_ydata([])

    return reward_plot


def _update_plot(plot, timestep, eval_reward):
    # Append values to axis then render
    plot.set_xdata(
        np.append(plot.get_xdata(), timestep))
    plot.set_ydata(
        np.append(plot.get_ydata(), eval_reward))
    plt.draw()
    plt.pause(0.01)
