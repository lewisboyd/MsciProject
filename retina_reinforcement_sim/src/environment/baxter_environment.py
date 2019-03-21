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


class BaxterEnvironment:
    """Class representing the learning environment."""

    def __init__(self, img_size=(256, 160)):
        """Initialise environment."""
        # Enable baxter and initialise environment
        baxter_interface.RobotEnable().enable()
        self._left_cam = CameraController("/cameras/left_hand_camera/image",
                                          img_size)
        self._left_arm = baxter_interface.Limb('left')
        self._left_arm.set_joint_position_speed(1.0)
        self._left_gripper = baxter_interface.Gripper('left')
        ns = "ExternalTools/left/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        self._move_to_start_position()
        self._load_gazebo_models()
        rospy.wait_for_service(ns, 5.0)

        # Observation information
        self.image = None
        self.img_size = img_size
        self.x_pos = None
        self.y_pos = None
        self.z_pos = None

        # State information
        self.horiz_dist = None
        self.vert_dist = None
        self.depth_dist = None

        # Maximum state values
        self._max_depth_dist = None

        # Accuracy tollerance
        self._horiz_threshold = 0.05
        self._vert_threshold = 0.05
        self._min_depth_threshold = 1
        self._max_depth_threshold = 2

        # Object position
        self._obj_x_pos = None
        self._obj_y_pos = None
        self._obj_z_pos = None

        # Limit on endpoint movements
        self._x_lim = 0.2
        self._y_lim = 0.3
        self._z_lim = 0.4

    def reset(self):
        """Reset arm position and move object to random location.

        Returns:
            observation and low state desciptor

        """
        self._move_to_start_position()
        # Loop until gripper not obscuring object
        visible = False
        while not visible:
            self._move_block()
            self._update_state()
            visible = self.horiz_dist is not None
        return [self.horiz_dist, self.vert_dist]

    def step(self, action):
        """Attempt to execute the action.

        Returns:
            observation, low state descriptor, reward, done

        """
        # Make move if possible and update observation/state
        joint_angles = self._get_joint_angles(action)
        if joint_angles:
            self._move_arm(joint_angles)
        self._update_state()

        # Calculate reward and if episode finished
        reward = 0.
        done = False
        if self.horiz_dist:
            # If object visible calculate reward based on distances
            reward -= abs(self.horiz_dist)
            reward -= abs(self.vert_dist)
            # if self.depth_dist < self._min_depth_threshold:
            #     reward -= 1
            # else:
            #     reward -= ((self.depth_dist - self.__min_depth_threshold)
            #                / self._max_depth_dist)

            # If sufficiently accurate then episode is finished
            # if (self.vert_dist < self._vert_threshold
            #         and self.horiz_dist < self._horiz_threshold
            #         and (self.depth_dist > self._min_depth_threshold
            #              and self.depth_dist < self._max_depth_dist)):
            #     done = True

            # If accurate enough then end the episode
            # if (abs(self.vert_dist) < self._vert_threshold
            #         and abs(self.horiz_dist) < self._horiz_threshold):
            #     done = True
        else:
            # If lost sight of object then end episode and return heavy penalty
            reward = -100
            done = True

        return [self.horiz_dist, self.vert_dist], reward, done

    def close(self):
        """Delete loaded models. To be called when node is shutting down."""
        self._delete_gazebo_models()

    def _update_state(self):
        # Update image
        self.image = self._left_cam.get_image()

        # Update cartesian coordinates
        pose = self._left_arm.endpoint_pose()
        self.x_pos = pose['position'].x
        self.y_pos = pose['position'].y
        self.z_pos = pose['position'].z
        # euler = self._get_euler(
        #     pose['orientation'])
        # euler = euler_from_quaternion(pose['orientation'])
        # self.roll_pos = euler[0]
        # self.pitch_pos = euler[1]
        # self.yaw_pos = euler[2]

        # Calculate distance to object and distance from centre of camera
        self.horiz_dist, self.vert_dist = self._calc_dist()
        # self.depth_dist = (self.x_pos - self._obj_x_pos) ** 2
        # + (self.y_pos - self._obj_y_pos) ** 2
        # + (self.z_pos - self._obj_z_pos) ** 2

    def _calc_dist(self):
        # Define lower and upper colour bounds
        BLUE_MIN = np.array([100, 150, 0], np.uint8)
        BLUE_MAX = np.array([140, 255, 255], np.uint8)

        # Threshold the image in the HSV colour space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, BLUE_MIN, BLUE_MAX)

        # If no object in image return none
        moments = cv2.moments(mask)
        if (moments["m00"] == 0):
            return None, None

        # Calculate normalised horizontal and vertical distances
        obj_x = int(moments["m10"] / moments["m00"])
        obj_y = int(moments["m01"] / moments["m00"])
        x_dist = obj_x - self.img_size[0] / 2
        y_dist = obj_y - self.img_size[1] / 2
        x_dist = float(x_dist) / (self.img_size[0] / 2)
        y_dist = float(y_dist) / (self.img_size[1] / 2)
        return x_dist, y_dist

    def _get_joint_angles(self, action):
        # Calculate desired endpoint position of arm
        x = self.x_pos + action[0] * self._x_lim
        y = self.y_pos + action[1] * self._y_lim
        # z = self.z_pos + action[2] * self._z_lim

        # Create pose of desired position
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.3
        pose.orientation.x = 0.0
        pose.orientation.y = 1.0
        pose.orientation.z = 0.0
        pose.orientation.w = 0.0

        # Return result from inverse kinematics or none if move invalid
        i = 0
        joint_angles = None
        while joint_angles is None and i < 5:
            joint_angles = self._ik_request(pose)
            i = i + 1
        return joint_angles

    def _move_to_start_position(self):
        # Move arm to the starting position
        starting_joint_angles = {'left_w0': 0.003,
                                 'left_w1': 1.379,
                                 'left_w2': -0.121,
                                 'left_e0': 0.020,
                                 'left_e1': 1.809,
                                 'left_s0': -0.935,
                                 'left_s1': -1.618}
        self._move_arm(starting_joint_angles)
        self._left_gripper.open()

    def _ik_request(self, pose):
        # Send request
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return None

        # If successful then return joint angles other return none
        resp_seeds = struct.unpack('<%dB' % len(
            resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            limb_joints = dict(
                zip(resp.joints[0].name, resp.joints[0].position))
        else:
            return None
        return limb_joints

    def _move_arm(self, joint_angles):
        # Gets differences between desired and actual joint angles
        def genf(joint, angle):
            def joint_diff():
                return abs(angle - self._left_arm.joint_angle(joint))
            return joint_diff
        diffs = [genf(j, a) for j, a in joint_angles.items()]

        # Move arm until in correct position
        self._left_arm.set_joint_positions(joint_angles, raw=True)
        baxter_dataflow.wait_for(
            lambda: (all(diff() < 0.008726646 for diff in diffs)),
            timeout=2.0,
            rate=100,
            raise_on_error=False,
            body=lambda: self._left_arm.set_joint_positions(joint_angles,
                                                            raw=True)
        )

    def _move_block(self):
        # Generate random x and y locations within reachable area
        x_loc = random.uniform(0.29, 0.68)
        y_loc = random.uniform(0.0, 0.38)

        # Create ModelState message
        block_pos = Pose(position=Point(x=x_loc, y=y_loc, z=0.7825))
        model_state = ModelState(
            model_name='block', pose=block_pos, reference_frame="world"
        )

        # Attempt to move block
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_model_state = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState
            )
            set_model_state(model_state)
            self._obj_x_pos = block_pos.position.x
            self._obj_y_pos = block_pos.position.y
            self._obj_z_pos = block_pos.position.z
        except rospy.ServiceException, e:
            rospy.logerr("Set Model State service call failed: {0}".format(e))

    def _load_gazebo_models(self):
        # Get Models' Path
        model_path = (
            rospkg.RosPack().get_path('retina_reinforcement_sim') + "/models/"
        )

        # Load Table SDF
        table_xml = ''
        with open(model_path + "cafe_table/model.sdf", "r") as table_file:
            table_xml = table_file.read().replace('\n', '')

        # Load block URDF
        block_xml = ''
        with open(model_path + "block/model.urdf", "r") as block_file:
            block_xml = block_file.read().replace('\n', '')

        # Spawn Table SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            pose = Pose(position=Point(x=0.700, y=0.213, z=0.000))
            reference_frame = "world"
            spawn_sdf = rospy.ServiceProxy(
                '/gazebo/spawn_sdf_model', SpawnModel)
            spawn_sdf("table", table_xml, "/", pose, reference_frame)
        except rospy.ServiceException, e:
            rospy.logerr("Spawn SDF service call failed: {0}".format(e))

        # Spawn Block URDF
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            pose = Pose(position=Point(x=0.485, y=0.190, z=0.7825))
            reference_frame = "world"
            spawn_urdf = rospy.ServiceProxy(
                '/gazebo/spawn_urdf_model', SpawnModel)
            spawn_urdf("block", block_xml, "/",
                       pose, reference_frame)
        except rospy.ServiceException, e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    def _delete_gazebo_models(self):
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model',
                                              DeleteModel)
            delete_model("block")
            delete_model("table")
        except rospy.ServiceException, e:
            rospy.loginfo("Model service call failed: {0}".format(e))
