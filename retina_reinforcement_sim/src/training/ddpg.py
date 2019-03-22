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
                 actor_optim, critic, critic_optim, preprocessor,
                 s_normalizer=None, r_normalizer=None):
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

        # Running mean and std normalization
        self.s_normalizer = s_normalizer
        self.r_normalizer = r_normalizer

    def train(self, env, init_explore, max_steps, max_ep_steps,
              updates_per_step, model_folder, result_folder, data_folder=None,
              eval_freq=5000, eval_ep=10, checkpoint=None):
        """Train the agent.

        Args:
            env (object): Environment to train agent in.
            init_explore (int): Number of timesteps to prepopulate replay
                                buffer.
            max_steps (int): Maximum number of training steps.
            max_ep_steps (int): Maximum number of steps in one episode.
            updates_per_step (int): How many updates to run per environment
                                    step.
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
            timestep_t = 0
            timestep_ep = 0
            done = False
            self.noise_function.reset()
            state = self.preprocessor(env.reset()).to(self.device)
            if self.s_normalizer:
                self.s_normalizer.observe(state)
            while timestep_t < init_explore:
                timestep_t = timestep_t + 1
                timestep_ep = timestep_ep + 1

                if done:
                    # Reset environment
                    self.noise_function.reset()
                    state = self.preprocessor(env.reset()).to(self.device)
                    if self.s_normalizer:
                        self.s_normalizer.observe(state)
                    timestep_ep = 0

                # Step through environment
                action = torch.tensor(self.noise_function(),
                                      device=self.device,
                                      dtype=torch.float).clamp(-1, 1)
                next_obs, reward, done = env.step(action.cpu())
                if self.r_normalizer:
                    self.r_normalizer.observe(reward)
                done = done or (timestep_ep == max_ep_steps)

                # Convert to tensors and save
                next_state = self.preprocessor(next_obs).to(self.device)
                reward = self._reward_to_tensor(reward)
                self.memory.push(state, action, next_state,
                                 reward, self._done_to_tensor(done))

                # Update current state
                state = next_state

        try:
            ep = 1
            eval_t = 0
            timestep_t = 0
            timestep_ep = 0
            ep_reward = 0.
            done = False
            q_loss = 0.0
            p_loss = 0.0
            ep_p_losses = []
            self.noise_function.reset()
            state = self.preprocessor(env.reset()).to(self.device)
            if self.s_normalizer:
                self.s_normalizer.observe(state)

            # Run training loop
            while timestep_t < max_steps:

                if done:
                    # Report episode performance
                    q_loss /= (timestep_ep * updates_per_step)
                    p_loss /= (timestep_ep * updates_per_step)
                    ep_q_losses.append(q_loss)
                    ep_p_losses.append(p_loss)
                    print "Timestep: %7d/%7d Episode: %4d Reward: %0.2f Critic Loss: %0.3f Actor Loss: %0.3f" % (
                        timestep_t, max_steps, ep, ep_reward, q_loss, p_loss)

                    # Reset the environment
                    self.noise_function.reset()
                    state = self.preprocessor(env.reset()).to(self.device)
                    if self.s_normalizer:
                        self.s_normalizer.observe(state)
                    timestep_ep = 0
                    ep_reward = 0.
                    q_loss = 0.0
                    p_loss = 0.0
                    ep = ep + 1

                if timestep_t == 0 or timestep_t % eval_t == 0:
                    # Evaluate performance
                    eval_reward = self._evaluate(
                        env, max_ep_steps, eval_ep)
                    validation_rewards.append(eval_reward)

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

                    # Reset the environment
                    self.noise_function.reset()
                    state = self.preprocessor(env.reset()).to(self.device)
                    if self.s_normalizer:
                        self.s_normalizer.observe(state)
                    timestep_ep = 0
                    ep_reward = 0.
                    q_loss = 0.0
                    p_loss = 0.0
                    eval_t += eval_freq

                # Step through environment
                timestep_t = timestep_t + 1
                timestep_ep = timestep_ep + 1
                action = self._get_exploration_action(state)
                next_obs, reward, done = env.step(action.cpu())
                if self.r_normalizer:
                    self.r_normalizer.observe(reward)
                done = done or (timestep_ep == max_ep_steps)
                ep_reward += reward

                # Convert to tensors and save
                next_state = self.preprocessor(next_obs).to(self.device)
                reward = self._reward_to_tensor(reward)
                self.memory.push(state, action, next_state,
                                 reward, self._done_to_tensor(done))

                # Optimise agent
                for _ in range(updates_per_step):
                    critic_loss, actor_loss = self._optimise()
                    q_loss += critic_loss
                    p_loss += actor_loss

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
                self._save(model_folder, str(max_steps))

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

        # Normalize
        if self.s_normalizer:
            state_batch = self.s_normalizer.normalize(state_batch)
            next_state_batch = self.s_normalizer.normalize(next_state_batch)
        if self.r_normalizer:
            reward_batch = self.r_normalizer.normalize(reward_batch)

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
        actor_loss = -(self.critic(state_batch,
                                   self.actor(state_batch))).mean()

        # Optimise actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update
        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)

        return critic_loss.item(), actor_loss.item(),

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
            if self.s_normalizer:
                state = self.s_normalizer.normalize(state)
            self.actor.eval()
            action = self.actor(state.unsqueeze(0)).squeeze(0)
            self.actor.train()
            return action

    def _get_exploration_action(self, state):
        with torch.no_grad():
            if self.s_normalizer:
                state = self.s_normalizer.normalize(state)

            noise_values = torch.tensor(
                self.noise_function(), device=self.device)
            if self.noise > self.final_noise:
                self.noise -= self.noise_decay
                if self.noise < self.final_noise:
                    self.noise = self.final_noise
            action = self.actor(state.unsqueeze(0)).squeeze(0)
            noisy_action = action + self.noise * noise_values.float()
            return noisy_action.clamp(-1, 1)

    def _save(self, path, id):
        torch.save(self.actor.state_dict(), path + id + "_actor")
        torch.save(self.critic.state_dict(), path + id + "_critic")
        if self.s_normalizer:
            self.s_normalizer.save(path + id)

    def _load(self, path, checkpoint):
        self.actor.load_state_dict(torch.load(path + checkpoint + "_actor"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(path + checkpoint + "_critic"))
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.s_normalizer:
            self.s_normalizer.load(path + checkpoint)

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
