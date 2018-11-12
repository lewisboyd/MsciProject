#!/usr/bin/env python

import cortex
import cortex_cuda
import retina_cuda
import sys
from collections import namedtuple
import math
from time import sleep
import random
from threading import Thread
from threading import Event
from Queue import Queue

import cv2
import numpy as np
import cPickle as pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(95040, 512)
        self.head = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 95040)
        x = F.relu(self.fc1(x))
        x = self.head(x)
        return x


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model():
    if len(memory) < 3:
        return
    transitions = memory.sample(3)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.stack(batch.action)
    print "Action batch"
    print action_batch
    next_state_batch = torch.cat(batch.next_state)
    reward_batch = torch.cat(batch.reward)
    print "Reward batch"
    print reward_batch

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch)
    print "State action values"
    print state_action_values

    # Compute V(s_{t+1})
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    print "Next state values"
    print next_state_values

    # Compute expected Q values
    expected_state_action_values = (next_state_values * 0.99) + reward_batch
    print "Expected state action values"
    print expected_state_action_values

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(5)
for i in range(5):
    memory.push(torch.rand(1, 3, 246, 468), torch.rand(3),
                torch.rand(1, 3, 246, 468), torch.rand(1))
optimize_model()
