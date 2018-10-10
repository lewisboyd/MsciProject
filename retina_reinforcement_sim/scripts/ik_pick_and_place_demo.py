#!/usr/bin/env python

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Inverse Kinematics Pick and Place Demo
"""
import argparse
import struct
import sys
import copy

import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface

class TrainDcnn(object):
    def __init__(self, limb):
        self._limb_name = limb
        self._limb = baxter_interface.Limb(limb)

        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

def load_gazebo_models(ball_pose=Pose(position=Point(x=5.0, y=0.0, z=2.0)),
                       ball_reference_frame="world"):

    # Get Models' Path
    model_path = rospkg.RosPack().get_path('retina_reinforcement_sim')+"/models/"

    # Load ball SDF
    ball_xml = ''
    with open (model_path + "simple_ball/model.sdf", "r") as ball_file:
        ball_xml=ball_file.read().replace('\n', '')

    # Spawn ball SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("simple_ball", ball_xml, "/",
                             ball_pose, ball_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))

def delete_gazebo_models():
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("simple_ball")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))

def main():
    rospy.init_node("ik_pick_and_place_demo")
    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    load_gazebo_models()
    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_models)

    # Wait for the All Clear from emulator startup
    rospy.wait_for_message("/robot/sim/started", Empty)

    limb = 'left'
    # Starting Joint angles for left arm
    starting_joint_angles = {'left_w0': 0.6699952259595108,
                             'left_w1': 1.030009435085784,
                             'left_w2': -0.4999997247485215,
                             'left_e0': -1.189968899785275,
                             'left_e1': 1.9400238130755056,
                             'left_s0': -0.08000397926829805,
                             'left_s1': -0.9999781166910306}
    pnp = TrainDcnn(limb)

    pnp.move_to_start(starting_joint_angles)

    rospy.spin()

if __name__ == '__main__':
    sys.exit(main())
