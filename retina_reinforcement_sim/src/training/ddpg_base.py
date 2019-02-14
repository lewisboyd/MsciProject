import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from memory import ReplayMemory


class DdpgBase:
    """Base class that implements training loop."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, actor, actor_args, critic,
                 critic_args, reward_scale):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int) : Maximum capacity of the memory
            batch_size (int) : Sample size from memory when optimising
            noise_function (object) : Function to generate random additive
                                      noise
            init_noise (double) : Initial amount of noise to be added
            final_noise (double) : Final amount of noise to be added
            exploration_len (int) : Number of steps to decay noise over
            actor (object): Actor constructor to use
            actor_args (array): Args for actor constructor
            critic (object): Critic constructor to use
            critic_args (array): Args for critic constructor
            reward_scale (float): Rescales received rewards
        """
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.noise_function = noise_function
        self.noise = init_noise
        self.final_noise = final_noise
        self.noise_decay = (init_noise - final_noise) / exploration_len
        self.tau = 0.001
        self.discount = 0.99
        self.reward_scale = reward_scale
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")

        # Create actor networks
        self.actor = actor(*actor_args).to(self.device)
        self.actor_target = actor(*actor_args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), 0.0001)

        # Create critic networks
        self.critic = critic(*critic_args).to(self.device)
        self.critic_target = critic(*critic_args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), 0.001,
                                             weight_decay=0.01)

    def interpret(self, obs):
        """Process environment obsevation to get state tensor."""
        raise NotImplementedError("Implement in child class.")

    def train(self, env, init_explore, max_episodes, max_steps,
              model_folder, data_folder, plot_ylim=[-200, 0], eval_freq=100,
              eval_ep=10):
        """Train the agent.

        Args:
            env (object): Environment to train agent in.
            init_explore (int): Number of episodes to prepopulate replay
                                buffer.
            max_episodes (int): Maximum number of training episodes.
            max_steps (int): Maximum number of steps in one episode.
            model_folder (path): Folder to save models in during training.
            data_folder (path): Folder to save evaluation data.
            plot_ylim (array): Min and max value for episode reward axis.
            eval_freq (int): How many episodes of training before next
                             evaluation.
            eval_ep (int): How many episodes to run for evaluation.
        """
        # Create folders for saving data
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)

        # Create interactive plot
        reward_plot = self._create_plot(max_episodes, max_steps, plot_ylim)

        start = time.time()

        # Populate replay buffer using noise function
        for _ in range(init_explore):
            # Reset noise function and environment
            self.noise_function.reset()
            obs = env.reset()
            state = self.interpret(obs)

            for step in range(max_steps):
                # Step using noise value
                action = torch.tensor(self.noise_function(),
                                      device=self.device,
                                      dtype=torch.float).clamp(-1, 1)
                new_obs, reward = env.step(action)
                done = step == max_steps - 1

                # Convert to tensors and save
                new_state = self.interpret(new_obs)
                reward = self._reward_to_tensor(reward)
                done = self._done_to_tensor(done)
                self.memory.push(state, action, new_state, reward, done)

                # Update current state
                state = new_state

        # Evaluate initial performance
        eval_reward = self._evaluate(env, max_steps, eval_ep)
        self._update_plot(reward_plot, 0, eval_reward)
        print "Initial Performance: %f" % eval_reward

        # Train models
        for ep in range(1, max_episodes + 1):
            # Reset noise function and environment
            self.noise_function.reset()
            obs = env.reset()
            state = self.interpret(obs)

            ep_reward = 0.
            for step in range(max_steps):
                # Step using noisey action
                action = self._get_exploration_action(state)
                new_obs, reward = env.step(action)
                done = step == max_steps - 1
                ep_reward += reward.item()

                # Convert to tensors and save
                new_state = self.interpret(new_obs)
                reward = self._reward_to_tensor(reward)
                done = self._done_to_tensor(done)
                self.memory.push(state, action, new_state, reward, done)

                # Optimise agent
                self._optimise()

                # Update current state
                state = new_state

            print "Episode: %4d/%4d  Reward: %0.2f" % (
                ep, max_episodes, ep_reward)

            # Evaluate performance and save agent every 100 episodes
            if ep % eval_freq == 0:
                eval_reward = self._evaluate(env, max_steps, eval_ep)
                self._update_plot(reward_plot, ep * max_steps, eval_reward)
                self._save(model_folder, str(ep))
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

    def _optimise(self):
        # Sample memory
        if len(self.memory) < self.batch_size:
            return None, None
        sample = self.memory.sample(self.batch_size)
        batch = ReplayMemory.Experience(*zip(*sample))
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.stack(batch.done)

        # Compute critic loss
        with torch.no_grad():
            next_q = self.critic_target(
                next_state_batch, self.actor_target(next_state_batch))
            expected_q = reward_batch + (self.discount * done_batch * next_q)
        predicted_q = self.critic.forward(state_batch, action_batch)
        critic_loss = F.mse_loss(predicted_q, expected_q)

        # Optimise critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_value_(self.critic.parameters(), 1)
        self.critic_optim.step()

        # Compute actor loss
        actor_loss = -self.critic(state_batch,
                                  self.actor(state_batch)).mean()

        # Optimise actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update
        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)

    def __reward_to_tensor(self, reward):
        return torch.tensor([reward * self.reward_scale],
                            dtype=torch.float).to(self.device)

    def _done_to_tensor(self, done):
        if done:
            return torch.tensor([0], dtype=torch.float).to(self.device)
        return torch.tensor([1], dtype=torch.float).to(self.device)

    def _evaluate(self, env, max_steps, eval_episodes=10):
        # Average multiple episode rewards
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = env.reset()
            state = self.interpret(obs)
            for step in range(max_steps):
                action = self._get_exploitation_action(state)
                new_obs, reward = env.step(action)
                avg_reward += reward
                state = self.interpret(obs)
        return avg_reward / eval_episodes

    def _get_exploitation_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            action = self.actor(state.unsqueeze(0)).view(1)
            self.actor.train()
            return action

    def _get_exploration_action(self, state):
        with torch.no_grad():
            noise_values = torch.tensor(
                self.noise_function(), device=self.device)
            # if self.noise > self.final_noise:
            #     self.noise -= self.noise_decay
            #     if self.noise < self.final_noise:
            #         self.noise = self.final_noise
            action = self.actor(state.unsqueeze(0)).view(1)
            noisy_action = action + self.noise * noise_values.float()
            return noisy_action.clamp(-1, 1)

    def _save(self, path, id):
        torch.save(self.actor.state_dict(), path + "actor_" + id)
        torch.save(self.critic.state_dict(), path + "critic_" + id)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau)
                                    + param.data * self.tau)

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
