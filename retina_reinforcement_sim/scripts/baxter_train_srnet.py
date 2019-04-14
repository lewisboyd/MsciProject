#!/usr/bin/env python

import os
import time
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from model import ResNet10, ResNet6, WRN64, WRN128


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
        split = states.size(0) - (states.size(0) / 5)
        self.start_index = 0
        if is_test:
            self.states = states[split:]
            self.start_index = split
        else:
            self.states = states[:split]

    def __len__(self):
        """Return length of dataset."""
        return self.states.size(0)

    def __getitem__(self, id):
        """Index dataset to get image and state pair."""
        img_name = self.img_dir + "img" + str(self.start_index + id) + ".png"
        img = cv2.imread(img_name)
        img = cv2.resize(img, None, fx=0.7, fy=0.7,
                         interpolation=cv2.INTER_AREA)
        # Convert to float and rescale to [0,1]
        img = img.astype(np.float32) / 255
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        state = self.states[id]
        return(img, state)


def str2bool(value):
    """Convert string to boolean."""
    return value.lower() == 'true'


def lower(value):
    """Returns lowercase string"""
    return value.lower()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train State Representation Net.')
    parser.add_argument('--use-retina', type=str2bool, required=True,
                        help='if true trains using retina images')
    parser.add_argument('--network', type=lower, required=True,
                        help='[ResNet6, ResNet10, WRN64, WRN128]')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(0)

    # Init network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.network == "resnet10":
        net = ResNet10(2).to(device)
    elif args.network == "resnet6":
        net = ResNet6(2).to(device)
    elif args.network == "wrn64":
        net = WRN64(2).to(device)
    elif args.network == "wrn128":
        net = WRN128(2).to(device)
    else:
        print "%s is not a valid network, choices are " % (
            args.network, "[ResNet6, ResNet10, WRN64, WRN128]")
        exit()
    optimiser = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimiser, 'min', factor=0.1, patience=10,
                                  min_lr=0.00001)

    # Training variables
    max_epoch = 50
    batch_size = 32
    criterion = nn.MSELoss()
    SAVE_FREQ = 5

    # Create save paths
    RESULTS_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/")
    MODEL_FOLDER = (os.path.dirname(
        os.path.realpath(__file__)) + "/baxter_center/")
    if args.use_retina:
        RESULTS_FOLDER = RESULTS_FOLDER + "retina_"
        MODEL_FOLDER = MODEL_FOLDER + "retina_"
    RESULTS_FOLDER = (RESULTS_FOLDER + args.network + "/results/")
    MODEL_FOLDER = (MODEL_FOLDER + args.network + "/state_dicts/")

    # Create folders if don't exist
    if not os.path.isdir(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if not os.path.isdir(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    # Create train and test dataloaders
    train_ds = ImageStateDataset(False, args.use_retina)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True,
                          num_workers=1, drop_last=True)
    test_ds = ImageStateDataset(True, args.use_retina)
    test_dl = DataLoader(test_ds, batch_size, pin_memory=True, num_workers=1,
                         drop_last=True)

    start = time.time()

    try:
        epoch_train_losses = []
        epoch_test_losses = []
        for epoch in range(1, max_epoch + 1):
            # Train
            net = net.train()
            train_losses = []
            for i, batch in enumerate(train_dl):
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)
                optimiser.zero_grad()
                preds = net(images)
                loss = criterion(preds, targets)
                loss.backward()
                optimiser.step()
                train_losses.append(loss.detach().cpu().numpy())

            # Test
            net = net.eval()
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

            # Update scheduler
            scheduler.step(test_loss)

            # Add train and test losses
            epoch_train_losses.append(train_loss)
            epoch_test_losses.append(test_loss)

            # Save model
            if (epoch % SAVE_FREQ == 0 and not epoch == max_epoch):
                torch.save(net.state_dict(), MODEL_FOLDER
                           + "net_" + str(epoch))
    finally:
        # Save results
        np.save(RESULTS_FOLDER + "train_loss", epoch_train_losses)
        np.save(RESULTS_FOLDER + "test_loss", epoch_test_losses)

        # Save model
        torch.save(net.state_dict(), MODEL_FOLDER
                   + "net_" + str(epoch))

        # Save training plot
        plt.figure()
        plt.plot(range(1, max_epoch + 1), epoch_train_losses)
        plt.plot(range(1, max_epoch + 1), epoch_test_losses)
        plt.legend(['Train loss', 'Test Loss'], loc='upper right')
        plt.savefig(RESULTS_FOLDER + "training_performance.png")

        end = time.time()
        mins = (end - start) / 60
        print "Training finished after %d hours %d minutes" % (
            mins / 60, mins % 60)
