import random
import struct

import numpy as np
import cv2
import rospy
import rospkg
import baxter_interface
import baxter_dataflow
from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
    SetModelState,
)
from gazebo_msgs.msg import ModelState
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
)
from std_msgs.msg import Header

from camera_controller import CameraController
from baxter_environment import BaxterEnvironment


class BaxterEnvironmentHer(BaxterEnvironment):
    """Class representing the learning environment."""

    def __init__(self, img_size=(256, 160)):
        """Initialise environment."""
        BaxterEnvironment.__init__(self, img_size)

        # Accuracy tollerance
        self._horiz_threshold = 0.05
        self._vert_threshold = 0.05
        self._min_depth_threshold = 1
        self._max_depth_threshold = 2

    def get_reward(self, state, goal):
        if state[0] is None:
            return -1
        horiz_diff = abs(state[0] - goal[0])
        vert_diff = abs(state[1] - goal[1])
        if (horiz_diff < self._horiz_threshold
                and vert_diff < self._vert_threshold):
            return 0
        return -1

    def step(self, action):
        """Attempt to execute the action.

        Returns:
            state, reward, done

        """
        # Make move if possible and update observation/state
        joint_angles = self._get_joint_angles(action)
        if joint_angles:
            self._move_arm(joint_angles)
        self._update_state()

        # Calculate reward and if new episode should be started
        done = False
        if self.horiz_dist is None:
            # Lost sight of object then end episode
            done = True

        state = self._get_state()['state']
        return self._get_state(), self.get_reward(state, [0.0, 0.0]), done

    def _get_state(self):
        state = [self.horiz_dist, self.vert_dist, self.x_pos, self.y_pos]
        desired_goal = [0.0, 0.0]
        achieved_goal = [self.horiz_dist, self.vert_dist]
        obs = {'state': state,
               'desired_goal': desired_goal,
               'achieved_goal': achieved_goal}
        return obs
