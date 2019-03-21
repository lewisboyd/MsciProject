#!/usr/bin/env python

import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from environment import BaxterEnvironment
from model import (ActorMlp, CriticMlp)
from training import (Ddpg, BaxterPreprocessor, NormalActionNoise)


if __name__ == '__main__':
    # Get folders
    folders = sys.argv[1:]

    if len(folders) == 1:
        # Load data
        folder = folders[0]
        ep_q_losses = np.load(folder + "/q_loss.npy")
        ep_p_losses = np.load(folder + "/p_loss.npy")
        validation_rewards = np.load(folder + "/ep_reward.npy")

        # Plot data
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(validation_rewards)
        ax[1].plot(ep_q_losses)
        ax[2].plot(ep_p_losses)

        # Set titles
        ax[0].set_title("Evaluation rewards")
        ax[1].set_title("Critic loss")
        ax[2].set_title("Actor loss")

        # Remove x labels
        ax[0].get_xaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        ax[2].get_xaxis().set_visible(False)

        # Display
        fig.tight_layout()
        plt.show()

    else:
        # Set common titles
        fig, ax = plt.subplots(len(folders), 3)
        ax[0, 0].set_title("Evaluation rewards")
        ax[0, 1].set_title("Critic loss")
        ax[0, 2].set_title("Actor loss")

        row = 0
        for folder in folders:
            # Get data
            ep_q_losses = np.load(folder + "/q_loss.npy")
            ep_p_losses = np.load(folder + "/p_loss.npy")
            validation_rewards = np.load(folder + "/ep_reward.npy")

            # Plot data
            ax[row, 0].plot(validation_rewards)
            ax[row, 1].plot(ep_q_losses)
            ax[row, 2].plot(ep_p_losses)

            # Remove x labels
            ax[row, 0].get_xaxis().set_visible(False)
            ax[row, 1].get_xaxis().set_visible(False)
            ax[row, 2].get_xaxis().set_visible(False)

            # Move to next row
            row = row + 1

        # Display
        fig.tight_layout()
        plt.show()
