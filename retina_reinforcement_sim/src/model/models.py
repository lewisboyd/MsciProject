import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Represents an Actor in the Actor to Critic Model.

    The Actor takes in a cortical image (state) and outputs 3 values between -1
    and 1 for the found_centre action, move_wrist_action and move_elbow action
    respectively.

    """

    def __init__(self, feature_extractor, num_features):
        """Initialise layers.

        Args:
            feature_extractor (model) : The model to use to extract image
                                        features
            num_features (int) : Number of features the extractor will return

        """
        super(Actor, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, 3)
        self.head = nn.Tanh()

    def forward(self, image):
        """Pass the image through the network.

        Args:
            image (float tensor) : RGB cortical image

        Returns:
            float tensor: found_centre, move_wrist and move_elbow policy

        """
        x = self.feature_extractor(image)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x


class Critic(nn.Module):
    """Represents a Critic in the Actor to Critic Model.

    The Critic takes in a cortical image (state) and the 3 action values to
    output the expected value of carrying out the action in the state.

    """

    def __init__(self, feature_extractor, num_features):
        """Initialise layers.

        Args:
            feature_extractor (model) : The model to use to extract image
                                        features
            num_features (int) : Number of features the extractor will return

        """
        super(Critic, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc1 = nn.Linear(num_features + 3, 512)
        self.head = nn.Linear(512, 1)

    def forward(self, image, action):
        """Pass the image and actions through the network.

        Args:
            image (float tensor) : RGB cortical image
            action (float tensor) : The 3 action values

        Returns:
            float tensor : value of the action values in the state

        """
        x = self.feature_extractor(image)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, action), 1)
        x = F.relu(self.fc1(x))
        x = self.head(x)
        return x
