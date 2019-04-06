#!/usr/bin/env python

import os
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model import FeatureExtractorNorm


class ImageStateDataset(Dataset):
    """Dataset of images and object location."""

    def __init__(self, is_test, use_retina):
        """Set base paths and load in states.

        Args:
            is_test (boolean): If True loads test set, else train set
            use_retina (boolean): If True uses retina images
        """
        self.base_dir = (os.path.dirname(os.path.realpath(__file__))
                         + "/baxter_center/image_data/")
        if use_retina:
            self.img_dir = self.base_dir + "retina_images/"
        else:
            self.img_dir = self.base_dir + "images/"
        states = torch.load(self.base_dir + "states")
        split = states.size(0) - (states.size(0) / 10)
        if is_test:
            self.states = states[split:]
        else:
            self.states = states[:split]

    def __len__(self):
        """Return length of dataset."""
        return self.states.size(0)

    def __getitem__(self, id):
        """Index dataset to get image and state pair."""
        img_name = self.img_dir + "img" + str(id) + ".png"
        img = cv2.imread(img_name)
        # Convert to float and rescale to [0,1]
        img = img.astype(np.float32) / 255
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        state = self.states[id]
        return(img, state)


def str2bool(value):
    """Convert string to boolean."""
    return value.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train State Representation Net.')
    parser.add_argument('--use-retina', type=str2bool, required=True,
                        help='if true trains using retina images')
    args = parser.parse_args()

    # Training variables
    use_retina = False
    max_epoch = 100
    batch_size = 32
    criterion = nn.MSELoss()
    SAVE_FREQ = 5
    RESULTS_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/")
    MODEL_FOLDER = (os.path.dirname(os.path.realpath(__file__))
                    + "/baxter_center/")
    if args.use_retina:
        RESULTS_FOLDER = RESULTS_FOLDER + "sr_retina/results/"
        MODEL_FOLDER = MODEL_FOLDER + "sr_retina/state_dicts/"
    else:
        RESULTS_FOLDER = RESULTS_FOLDER + "sr/results/"
        MODEL_FOLDER = MODEL_FOLDER + "sr/state_dicts/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.use_retina:
        net = FeatureExtractorNorm(3, 2).to(device)
    else:
        net = FeatureExtractorNorm(3, 2).to(device)
    optimiser = optim.Adam(net.parameters())

    # Create folders if don't exist
    if not os.path.isdir(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if not os.path.isdir(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    # Create train and test dataloaders
    train_ds = ImageStateDataset(False, args.use_retina)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True,
                          num_workers=4)
    test_ds = ImageStateDataset(True, args.use_retina)
    test_dl = DataLoader(test_ds, batch_size, pin_memory=True, num_workers=4)

    start = time.time()

    # try:
    epoch_train_losses = []
    epoch_test_losses = []
    for epoch in range(1, max_epoch + 1):
        # Train
        train_losses = []
        for i, batch in enumerate(train_dl):
            images, targets = batch
            print images.size()
            print targets.size()
            images = images.to(device)
            targets = targets.to(device)
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
                images = images.to(device)
                targets = targets.to(device)
                preds = net(images)
                loss = criterion(preds, targets)
                test_losses.append(loss.detach().cpu().numpy())

        # Report performance
        train_loss = np.asarray(train_losses).mean()
        test_loss = np.asarray(test_losses).mean()
        print "Epoch: %3d/%3d  Train Loss: %0.4f Test Loss: %0.4f" % (
            epoch, max_epoch, train_loss, test_loss)

        epoch_train_losses.append(train_loss)
        epoch_test_losses.append(test_loss)

        # Save model
        if (epoch % SAVE_FREQ == 0 and not epoch == max_epoch):
            torch.save(net.state_dict(), MODEL_FOLDER
                       + "net_" + str(epoch))
    # finally:
    #     # Plot training results
    #     plt.figure()
    #     plt.plot(range(1, max_epoch + 1), epoch_train_losses)
    #     plt.plot(range(1, max_epoch + 1), epoch_test_losses)
    #     plt.legend(['Train loss', 'Test Loss'], loc='upper right')
    #
    #     # Save results
    #     plt.savefig(RESULTS_FOLDER + "training_performance.png")
    #     np.save(RESULTS_FOLDER + "train_loss", epoch_train_losses)
    #     np.save(RESULTS_FOLDER + "test_loss", epoch_test_losses)
    #
    #     # Save model
    #     torch.save(net.state_dict(), MODEL_FOLDER
    #                + "net_" + str(epoch))
        #
        # end = time.time()
        # mins = (end - start) / 60
        # print "Training finished after %d hours %d minutes" % (
        #     mins / 60, mins % 60)
