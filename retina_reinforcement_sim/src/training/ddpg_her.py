import os
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from ddpg import Ddpg
from memory import HerMemory


class DdpgHer(Ddpg):
    """Base class that implements training loop."""

    def __init__(self, memory_capacity, batch_size, noise_function, init_noise,
                 final_noise, exploration_len, actor, actor_optim, critic,
                 critic_optim, s_normalizer=None):
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
            s_normalizer (object): Function to normalize observation
            r_normalizer (object): Function to normalize rewards
        """
        Ddpg.__init__(self, memory_capacity, batch_size, noise_function,
                      init_noise, final_noise, exploration_len, 1.0, actor,
                      actor_optim, critic, critic_optim, preprocessor=None,
                      s_normalizer=None, r_normalizer=None)

        # Experience Replay
        self.memory = HerMemory(memory_capacity)

    def train(self, env, init_explore, max_steps, max_ep_steps, model_folder,
              result_folder, data_folder=None, eval_freq=5000, eval_ep=10,
              checkpoint=None):
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
            eval_freq (int): How many timesteps of training before next
                             evaluation.
            eval_ep (int): How many episodes to run for evaluation.
            checkpoint (int): Checkpoint to restart training from
        """
        start = time.time()

        # Create folders for saving data
        if model_folder is not None and not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        if result_folder is not None and not os.path.isdir(result_folder):
            os.makedirs(result_folder)

        # Create arrays to store training data
        validation_rewards = []
        ep_q_losses = []
        ep_p_losses = []

        if data_folder is not None:
            # Prepopulate experience replay from data folder
            print "Loading data from folder: %s" % data_folder
            self._load_data(data_folder)

        if checkpoint:
            # Reload from checkpoint
            self._load(model_folder, checkpoint)

        if init_explore > 0:
            # Prepopulate experience replay using noise function
            print "Prepopulating experience replay"

            # Reset envionrment
            timestep_ep = 0
            done = False
            self.noise_function.reset()
            obs = env.reset()
            state = obs['state'].to(self.device)
            desired_goal = obs['desired_goal'].to(self.device)
            achieved_goal = obs['achieved_goal'].to(self.device)

            # Update running normalizer
            # if self.s_normalizer:
            #     self.s_normalizer.observe(torch.cat([state, goal]))

            # Store episode experiences
            ep_states = []
            ep_actions = []
            ep_next_states = []
            ep_rewards = []
            ep_dones = []

            timestep_t = 0
            while timestep_t < init_explore:
                timestep_t = timestep_t + 1
                timestep_ep = timestep_ep + 1

                if done:
                    # Add episode experiences with desired goal
                    self.memory.push(ep_states, ep_actions, ep_next_states,
                                     ep_rewards, ep_dones, desired_goal)

                    # Add episode experiences with achieved goal and rewards
                    for i in range(len(ep_rewards)):
                        ep_rewards[i] = self._reward_to_tensor(env.get_reward(ep_next_states[i],
                                                       achieved_goal))
                    self.memory.push(ep_states, ep_actions, ep_next_states,
                                     ep_rewards, ep_dones, achieved_goal)

                    # Reset environment
                    timestep_ep = 0
                    done = False
                    self.noise_function.reset()
                    obs = env.reset()
                    state = obs['state'].to(self.device)
                    desired_goal = obs['desired_goal'].to(self.device)
                    achieved_goal = obs['achieved_goal'].to(self.device)
                    # if self.s_normalizer:
                    #     self.s_normalizer.observe(state)

                    # Clear episode experiences
                    ep_states = []
                    ep_actions = []
                    ep_next_states = []
                    ep_rewards = []
                    ep_dones = []

                # Step through environment
                action = torch.tensor(self.noise_function(),
                                      device=self.device,
                                      dtype=torch.float).clamp(-1, 1)
                next_obs, reward, done = env.step(action.cpu())
                if not done:
                    # If not ended episode early update achieved goal
                    achieved_goal = next_obs['achieved_goal'].to(self.device)
                done = done or (timestep_ep == max_ep_steps)

                # Convert to tensors
                next_state = next_obs['state'].to(self.device)
                reward = self._reward_to_tensor(reward)

                ep_states.append(state)
                ep_actions.append(action)
                ep_next_states.append(next_state)
                ep_rewards.append(reward)
                ep_dones.append(self._done_to_tensor(done))

                # Update current state
                state = next_state

        timestep_t = 0
        try:
            ep = 1
            eval_t = eval_freq

            # Reset environment
            q_loss = 0.0
            p_loss = 0.0
            timestep_ep = 0
            ep_reward = 0.
            done = False
            self.noise_function.reset()
            obs = env.reset()
            state = obs['state'].to(self.device)
            desired_goal = obs['desired_goal'].to(self.device)
            achieved_goal = obs['achieved_goal'].to(self.device)

            # Run training loop
            while timestep_t < max_steps:

                if done:
                    # Add episode experiences with desired goal
                    self.memory.push(ep_states, ep_actions, ep_next_states,
                                     ep_rewards, ep_dones, desired_goal)

                    # Add episode experiences with achieved goal and rewards
                    for i in range(len(ep_rewards)):
                        ep_rewards[i] = self._reward_to_tensor(env.get_reward(ep_next_states[i],
                                                       achieved_goal))
                    self.memory.push(ep_states, ep_actions, ep_next_states,
                                     ep_rewards, ep_dones, achieved_goal)

                    # Optimise agent
                    for _ in range(timestep_ep):
                        if timestep_t % eval_t == 0:
                            # Evaluate performance
                            eval_reward = self._evaluate(
                                env, max_ep_steps, eval_ep)
                            validation_rewards.append(eval_reward)
                            eval_t += eval_freq

                            # Report performance
                            print "Evaluation Reward: %0.2f" % eval_reward

                            if (model_folder is not None
                                    and not timestep_t == max_steps):
                                # Save model
                                self._save(model_folder, str(eval_t))
                            if (result_folder is not None
                                    and not timestep_t == max_steps):
                                # Save training data
                                np.save(result_folder + "q_loss", ep_q_losses)
                                np.save(result_folder + "p_loss", ep_p_losses)
                                np.save(result_folder + "ep_reward",
                                        validation_rewards)

                        if timestep_t > max_steps:
                            # If passed max steps then end training
                            break
                        else:
                            critic_loss, actor_loss = self._optimise()
                            q_loss += critic_loss
                            p_loss += actor_loss
                            timestep_t = timestep_t + 1

                    # Report episode performance
                    q_loss /= (timestep_ep)
                    p_loss /= (timestep_ep)
                    ep_q_losses.append(q_loss)
                    ep_p_losses.append(p_loss)
                    print "Timestep: %7d/%7d Episode: %4d Reward: %0.2f Critic Loss: %0.3f Actor Loss: %0.3f" % (
                        timestep_t, max_steps, ep, ep_reward, q_loss, p_loss)
                    ep = ep + 1

                    # Reset environment
                    q_loss = 0.0
                    p_loss = 0.0
                    timestep_ep = 0
                    ep_reward = 0.
                    done = False
                    self.noise_function.reset()
                    obs = env.reset()
                    state = obs['state'].to(self.device)
                    desired_goal = obs['desired_goal'].to(self.device)
                    achieved_goal = obs['achieved_goal'].to(self.device)

                    # Clear episode experiences
                    ep_states = []
                    ep_actions = []
                    ep_next_states = []
                    ep_rewards = []
                    ep_dones = []

                # Step through environment
                timestep_ep = timestep_ep + 1
                action = self._get_exploration_action(state, desired_goal)
                next_obs, reward, done = env.step(action.cpu())
                if not done:
                    # If not ended episode early update achieved goal
                    achieved_goal = next_obs['achieved_goal'].to(self.device)
                done = done or (timestep_ep == max_ep_steps)
                ep_reward += reward

                # Convert to tensors
                next_state = next_obs['state'].to(self.device)
                reward = self._reward_to_tensor(reward)

                # Add experiences
                ep_states.append(state)
                ep_actions.append(action)
                ep_next_states.append(next_state)
                ep_rewards.append(reward)
                ep_dones.append(self._done_to_tensor(done))

                # Update current state
                state = next_state
        finally:
            try:
                # Evaluate final performance
                eval_reward = self._evaluate(env, max_ep_steps, eval_ep)
                validation_rewards.append(eval_reward)
                print "Final Performance: %0.2f" % eval_reward
            except:
                pass

            if result_folder is not None:
                # Save all training data
                np.save(result_folder + "q_loss", ep_q_losses)
                np.save(result_folder + "p_loss", ep_p_losses)
                np.save(result_folder + "ep_reward", validation_rewards)

            if model_folder is not None:
                # Save the model
                self._save(model_folder, str(timestep_t))

            try:
                # Close environment
                env.close()
            except:
                pass

            # Report training time
            end = time.time()
            mins = (end - start) / 60
            print "Training finished after %d hours %d minutes" % (
                mins / 60, mins % 60)

    def _evaluate(self, env, max_steps, eval_episodes):
        # Average multiple episode rewards
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = env.reset()
            state = obs['state'].to(self.device)
            desired_goal = obs['desired_goal'].to(self.device)
            done = False
            timestep = 0
            while not done:
                timestep = timestep + 1
                action = self._get_exploitation_action(state, desired_goal)
                next_obs, reward, done = env.step(action.cpu())
                avg_reward += reward
                done = done or (timestep == max_steps)
                state = next_obs['state'].to(self.device)
        return avg_reward / eval_episodes

    def _get_exploitation_action(self, state, goal):
        with torch.no_grad():
            # if self.s_normalizer:
            #     state = self.s_normalizer.normalize(state)
            self.actor.eval()
            action = self.actor(torch.cat([state, goal]).unsqueeze(0)).squeeze(0)
            self.actor.train()
            return action

    def _get_exploration_action(self, state, goal):
        with torch.no_grad():
            # if self.s_normalizer:
            #     state = self.s_normalizer.normalize(state)

            noise_values = torch.tensor(
                self.noise_function(), device=self.device)
            if self.noise > self.final_noise:
                self.noise -= self.noise_decay
                if self.noise < self.final_noise:
                    self.noise = self.final_noise
            action = self.actor(torch.cat([state, goal]).unsqueeze(0)).squeeze(0)
            noisy_action = action + self.noise * noise_values.float()
            return noisy_action.clamp(-1, 1)
