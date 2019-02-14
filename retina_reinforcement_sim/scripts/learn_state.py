#!/usr/bin/env python

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from model import FeatureExtractor


if __name__ == '__main__':
    # Training variables
    max_epoch = 100
    batch_size = 1024
    criterion = nn.MSELoss()
    SAVE_FREQ = 10
    RESULTS_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/results/pixel_supervised/")
    MODEL_FOLDER = (os.path.dirname(os.path.realpath(__file__))
                    + "/pendulum/state_dicts/pixel_supervised/")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create folders if don't exist
    if not os.path.isdir(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if not os.path.isdir(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    # Load data
    DATA_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/pendulum/data/")
    images = torch.load(DATA_FOLDER + "images")
    targets = torch.load(DATA_FOLDER + "states")

    # Create train and test dataloaders
    test_split = images.size(0) - (images.size(0) / 10)
    train_ds = TensorDataset(images[:test_split], targets[:test_split])
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_ds = TensorDataset(images[test_split:], targets[test_split:])
    test_dl = DataLoader(test_ds, batch_size)

    # Create model
    net = FeatureExtractor(3, 3).to(device)
    optimiser = optim.Adam(net.parameters())

    epoch_train_losses = []
    epoch_test_losses = []
    for epoch in range(1, max_epoch + 1):
        # Train
        train_losses = []
        for i, batch in enumerate(train_dl):
            images, targets = batch
            optimiser.zero_grad()
            preds = net(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimiser.step()
            train_losses.append(loss.detach().cpu().numpy())

        # Test
        test_losses = []
        with torch.no_grad():
            for i, batch in enumerate(test_dl):
                images, targets = batch
                preds = net(images)
                loss = criterion(preds, targets)
                test_losses.append(loss)

        # Report performance
        train_loss = np.asarray(train_losses).mean()
        test_loss = np.asarray(test_losses).mean()
        print "Epoch: %3d/%3d  Train Loss: %0.2f Test Loss: %0.2f" % (
            epoch, max_epoch, train_loss, test_loss)

        epoch_train_losses.append(train_loss)
        epoch_test_losses.append(test_loss)

        # Save model
        if (epoch % SAVE_FREQ == 0 or epoch == max_epoch):
            torch.save(net.state_dict(), MODEL_FOLDER + "net_" + str(epoch))

    # Plot training results
    plt.figure()
    plt.plot(range(1, max_epoch + 1), epoch_train_losses)
    plt.plot(range(1, max_epoch + 1), epoch_test_losses)
    plt.legend(['Train loss', 'Test Loss'], loc='upper right')

    # Save results
    plt.savefig(RESULTS_FOLDER + "training_performance.png")
    np.asarray(epoch_train_losses).tofile(RESULTS_FOLDER + "train_losses.txt")
    np.asarray(epoch_test_losses).tofile(RESULTS_FOLDER + "test_losses.txt")
