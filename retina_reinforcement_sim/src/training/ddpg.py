import os
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from memory import ReplayMemory


class Ddpg:
    """Base class that implements training loop."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, reward_scale, actor,
                 actor_optim, critic, critic_optim, preprocessor):
        """Initialise networks, memory and training params.

        Args:
            memory_capacity (int) : Maximum capacity of the memory
            batch_size (int) : Sample size from memory when optimising
            noise_function (object) : Function to generate random additive
                                      noise
            init_noise (double) : Initial amount of noise to be added
            final_noise (double) : Final amount of noise to be added
            exploration_len (int) : Number of steps to decay noise over
            reward_scale (float): Rescales received rewards
            actor (object): Actor network
            actor_optim (array): Optimiser for actor
            critic (object): Critic network
            critic_optimm (array): Optimiser for critic
            preprocessor (object): Function to process environment observation
        """
        # Experience Replay
        self.memory = ReplayMemory(memory_capacity)

        # Noise function and variables
        self.noise_function = noise_function
        self.noise = init_noise
        self.final_noise = final_noise
        self.noise_decay = (init_noise - final_noise) / exploration_len

        # Training variables
        self.batch_size = batch_size
        self.tau = 0.001
        self.discount = 0.99
        self.reward_scale = reward_scale

        # Function to process environment observation before use by models
        self.preprocessor = preprocessor

        # Device to run on
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else
                                   "cpu")

        # Actor networks and optimiser
        self.actor = actor
        self.actor_target = copy.deepcopy(actor).to(self.device)
        self.actor_optim = actor_optim

        # Critic networks and optimiser
        self.critic = critic
        self.critic_target = copy.deepcopy(critic).to(self.device)
        self.critic_optim = critic_optim

    def train(self, env, init_explore, max_steps, max_ep_steps,
              model_folder, result_folder, data_folder=None,
              plot_ylim=[-200, 0], eval_freq=1000, eval_ep=10):
        """Train the agent.

        Args:
            env (object): Environment to train agent in.
            init_explore (int): Number of timesteps to prepopulate replay
                                buffer.
            max_steps (int): Maximum number of training steps.
            max_ep_steps (int): Maximum number of steps in one episode.
            model_folder (path): Folder to save models in during training.
            result_folder (path): Folder to save evaluation data.
            data_folder (path): If specified load to populate replay buffer.
            plot_ylim (array): Min and max value for episode reward axis.
            eval_freq (int): How many episodes of training before next
                             evaluation.
            eval_ep (int): How many episodes to run for evaluation.
        """
        # Create folders for saving data
        if model_folder is not None and not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        if result_folder is not None and not os.path.isdir(result_folder):
            os.makedirs(result_folder)

        # Create interactive plot
        reward_plot = self._create_plot(max_steps, plot_ylim)

        start = time.time()

        # If given a data folder prepopulate the experience replay
        if data_folder is not None:
            print "Loading data from folder: %s" % data_folder
            self._load_data(data_folder)

        # Populate replay buffer using noise function
        if init_explore > 0:
            print "Prepopulating experience replay"
            timestep_t = 0
            timestep_ep = 0
            done = False
            self.noise_function.reset()
            state = self.preprocessor(env.reset()).to(self.device)
            while timestep_t < init_explore:
                timestep_t = timestep_t + 1
                timestep_ep = timestep_ep + 1

                # When episode finished reset environment and noise function
                if done:
                    self.noise_function.reset()
                    state = self.preprocessor(env.reset()).to(self.device)
                    timestep_ep = 0

                # Step through environment
                action = torch.tensor(self.noise_function(),
                                      device=self.device,
                                      dtype=torch.float).clamp(-1, 1)
                next_obs, reward, done = env.step(action.cpu())
                done = done or (timestep_ep == max_ep_steps)

                # Convert to tensors and save
                next_state = self.preprocessor(next_obs).to(self.device)
                reward = self._reward_to_tensor(reward)
                self.memory.push(state, action, next_state,
                                 reward, self._done_to_tensor(done))

                # Update current state
                state = next_state

        # Evaluate initial performance
        print "Evaluating initial performance"
        eval_reward = self._evaluate(env, max_ep_steps, eval_ep)
        self._update_plot(reward_plot, 0, eval_reward)
        print "Initial Performance: %0.2f" % eval_reward

        # Run training loop
        try:
            ep = 1
            timestep_t = 0
            timestep_ep = 0
            timestep_eval = eval_freq
            ep_reward = 0.
            done = False
            self.noise_function.reset()
            state = self.preprocessor(env.reset()).to(self.device)
            timestep_ep = 0
            ep_reward = 0.
            while timestep_t < max_steps:
                if done:
                    # Report episode performance
                    print "Episode: %4d  Reward: %0.2f" % (
                        ep, ep_reward)
                    ep = ep + 1

                    # Reset the environment
                    self.noise_function.reset()
                    state = self.preprocessor(env.reset()).to(self.device)
                    timestep_ep = 0
                    ep_reward = 0.

                    # Run evaluation if enough timesteps have passed
                    if timestep_t >= timestep_eval:
                        eval_reward = self._evaluate(
                            env, max_ep_steps, eval_ep)
                        self._update_plot(reward_plot, ep
                                          * max_steps, eval_reward)
                        if (model_folder is not None
                                and not timestep_t == max_steps):
                            self._save(model_folder, str(timestep_eval))
                        print "Evaluation Reward: %0.2f" % eval_reward
                        timestep_eval = timestep_eval + eval_freq

                # Step through environment
                timestep_t = timestep_t + 1
                timestep_ep = timestep_ep + 1
                action = self._get_exploration_action(state)
                next_obs, reward, done = env.step(action.cpu())
                done = done or (timestep_ep == max_ep_steps)
                ep_reward += reward

                # Convert to tensors and save
                next_state = self.preprocessor(next_obs).to(self.device)
                reward = self._reward_to_tensor(reward)
                self.memory.push(state, action, next_state,
                                 reward, self._done_to_tensor(done))

                # Optimise agent
                self._optimise()

                # Update current state
                state = next_state
        finally:
            # Evaluate final performance
            print "Evaluating final performance"
            eval_reward = self._evaluate(env, max_ep_steps, eval_ep)
            self._update_plot(reward_plot, str(max_steps), eval_reward)
            print "Final Performance: %0.2f" % eval_reward

            if result_folder is not None:
                # Save figure and data
                plt.savefig(result_folder + "training_performance.png")
                reward_plot.get_xdata().tofile(result_folder
                                               + "eval_timesteps.txt")
                reward_plot.get_ydata().tofile(result_folder
                                               + "eval_rewards.txt")

            # Save the model
            if model_folder is not None:
                self._save(model_folder, str(max_steps))

            # Close figure and environment
            plt.clf()
            env.close()

            # Report training time
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

    def _reward_to_tensor(self, reward):
        return torch.tensor([reward * self.reward_scale],
                            dtype=torch.float).to(self.device)

    def _done_to_tensor(self, done):
        if done:
            return torch.tensor([0], dtype=torch.float).to(self.device)
        return torch.tensor([1], dtype=torch.float).to(self.device)

    def _evaluate(self, env, max_steps, eval_episodes):
        # Average multiple episode rewards
        avg_reward = 0.
        for _ in range(eval_episodes):
            state = self.preprocessor(env.reset()).to(self.device)
            done = False
            timestep = 0
            while not done:
                timestep = timestep + 1
                action = self._get_exploitation_action(state)
                next_obs, reward, done = env.step(action.cpu())
                avg_reward += reward
                done = done or (timestep == max_steps)
                state = self.preprocessor(next_obs).to(self.device)
        return avg_reward / eval_episodes

    def _get_exploitation_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            # action = self.actor(state.unsqueeze(0)).view(1)
            action = self.actor(state.unsqueeze(0)).squeeze(0)
            self.actor.train()
            return action

    def _get_exploration_action(self, state):
        with torch.no_grad():
            noise_values = torch.tensor(
                self.noise_function(), device=self.device)
            if self.noise > self.final_noise:
                self.noise -= self.noise_decay
                if self.noise < self.final_noise:
                    self.noise = self.final_noise
            # action = self.actor(state.unsqueeze(0)).view(1)
            action = self.actor(state.unsqueeze(0)).squeeze(0)
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

    def _load_data(self, data_folder):
        if (self.preprocessor.__class__.__name__ == 'PendulumPreprocessor'
                or self.preprocessor.__class__.__name__ == 'BaxterPreprocessor'):
            states = torch.load(data_folder + "states")
            next_states = torch.load(data_folder + "next_states")
        elif self.preprocessor.__class__.__name__ == 'ImagePreprocessor':
            states = torch.load(data_folder + "images")
            next_states = torch.load(data_folder + "next_images")
        elif self.preprocessor.__class__.__name__ == 'RetinaPreprocessor':
            states = torch.load(data_folder + "retina_images")
            next_states = torch.load(data_folder + "next_retina_images")
        actions = torch.load(data_folder + "actions")
        rewards = torch.load(data_folder + "rewards") * self.reward_scale
        dones = torch.load(data_folder + "dones")
        for i in range(states.size(0)):
            self.memory.push(states[i], actions[i], next_states[i],
                             rewards[i], dones[i])

    def _create_plot(self, max_steps, ylim):
        # Create interactive plot
        plt.ion()
        plt.show(block=False)
        ax = plt.subplot(111)
        ax.set_title('Evaluation Performance')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Episode Reward')

        # Set limits
        ax.set_xlim(0, max_steps)
        ax.set_ylim(ylim)

        # Create subplot without any data
        reward_plot, = ax.plot(0, 0, 'b')
        reward_plot.set_xdata([])
        reward_plot.set_ydata([])

        return reward_plot

    def _update_plot(self, plot, timestep, eval_reward):
        # Append values to axis then render
        plot.set_xdata(
            np.append(plot.get_xdata(), timestep))
        plot.set_ydata(
            np.append(plot.get_ydata(), eval_reward))
        plt.draw()
        plt.pause(0.01)
