import math
import random

import rospy
import rospkg
import baxter_interface
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
    Quaternion,
)
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

import image_processor
from camera_controller import CameraController

"""
TODO:
    normalise endpoint position
    normalise distances

    decide on all threshold values
    decide on all movement limit values

    load in environment

    test environment
"""

class Environment:
    """Class representing the learning environment."""

    def __init__(self, img_size=(160, 256)):
        """Initialise environment."""
        # Enable baxter and load environment models
        baxter_interface.RobotEnable().enable()
        self._load_gazebo_models()

        # Observation information
        self.image = None
        self.img_size = img_size
        self.x_pos = None
        self.y_pos = None
        self.z_pos = None
        self.roll_pos = None
        self.pitch_pos = None
        self.yaw_pos = None

        # State information
        self.horiz_dist = None
        self.vert_dist = None
        self.depth_dist = None

        # Maximum state values
        self._max_horiz_dist = img_size[1] / 2
        self._max_vert_dist = img_size[0] / 2
        self._max_depth_dist = None

        # Accuracy tollerance
        self._horiz_threshold = 10
        self._vert_threshold = 10
        self._min_depth_threshold = 1
        self._max_depth_threshold = 2

        # Object position
        self._obj_x_pos = None
        self._obj_y_pos = None
        self._obj_z_pos = None

        # Baxter interfaces
        self._left_cam = CameraController("/cameras/left_hand_camera/image",
                                          img_size)
        self._left_arm = baxter_interface.Limb('left')
        ns = "ExternalTools/left/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)

        # Limit on endpoint movements
        self._x_lim = None
        self._y_lim = None
        self._z_lim = None
        self._roll_lim = 0.79
        self._pitch_lim = 0.79
        self._yaw_lim = 0.79


    def reset(self):
        """Reset arm position and move object to random location.

        Returns:
            observation and low state desciptor
        """
        self._move_to_start_position()
        # Loop until gripper not obscuring object
        visible = False
        while not visible:
            self._move_object_location()
            self._update_state()
            visible = self.horiz_dist != -1
        return self._get_obs(), self._get_state()

    def step(self, action):
        """Attempts to execute the action.

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
            reward -= self.horiz_dist / self._max_horiz_dist
            reward -= self.vert_dist / self._max_vert_dist
            if self.depth_dist < self._min_depth_threshold:
                reward -= 1
            else:
                reward -= ((self.depth_dist - self.__min_depth_threshold)
                            / self._max_depth_dist)

            # If sufficiently accurate then episode is finished
            if (self.vert_dist < self._vert_threshold and
                self.horiz_dist < self._horiz_threshold and
                (self.depth_dist > self._min_depth_threshold and
                self.depth_dist < self._max_depth_dist)):
                done = True
        else:
            # If lost sight of object then end episode and return heavy penalty
            reward = -100
            done = True

        return self._get_obs(), self._get_state(), reward, done

    def shutdown(self):
        """Delete loaded models. To be called when node is shutting down."""
        self._delete_gazebo_models()

    def _update_state(self):
        # Update image
        self.image = self._left_cam.get_image()

        # Update caretesian coordinates
        pose = self._left_arm.endpoint_pose()
        self.x_pos = pose.position.x
        self.y_pos = pose.position.y
        self.z_pos = pose.position.z
        self.roll_pos, self.pitch, self.yaw = self._get_euler(pose.orientation)

        # Calculate distance to object and distance from centre of camera
        self.horiz_dist, self.vert_dist = self._calc_dist()
        self.depth_dist = (self.x_pos - self._obj_x_pos) ** 2
                          + (self.y_pos - self._obj_y_pos) ** 2
                          + (self.z_pos - self._obj_z_pos) ** 2

    def _calc_dist(self):
        # Define lower and upper colour bounds
        ORANGE_MIN = np.array([5, 50, 50], np.uint8)
        ORANGE_MAX = np.array([15, 255, 255], np.uint8)

        # Threshold the image in the HSV colour space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, ORANGE_MIN, ORANGE_MAX)

        # Calculate horizontal and vertical distance to centre if possible
        moments = cv2.moments(mask)
        if (moments["m00"] == 0):
            return -1, -1
        x_centre = int(moments["m10"] / moments["m00"])
        y_centre = int(moments["m01"] / moments["m00"])
        x_dist = self.img_size[1] - x_centre
        y_dist = self.img_size[0] - y_centre
        return x_dist, y_dist

    def _get_joint_angles(self, action):
        # Calculate desired endpoint position of arm
        x = self.x_pos + action[0] * self._x_lim
        y = self.y_pos + action[1] * self._y_lim
        z = self.z_pos + action[2] * self._z_lim
        roll = self.roll_pos + action[3] * self._roll_lim
        pitch = self.pitch_pos + action[4] * self._pitch_lim
        yaw = self.yaw_pos + action[5] * self._yaw_lim
        orientation = self._get_orientation(roll, pitch, yaw)

        # Create pose of desired position
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]

        # Return result from inverse kinematics
        return self._ik_request(pose)

    def _get_orientation(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

    def _get_euler(self, orientation):
        t0 = +2.0 * (orientation.w * orientation.x + orientation.y * orientation.z)
        t1 = +1.0 - 2.0 * (orientation.x * orientation.x + orientation.y * orientation.y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (orientation.w * orientation.y - orientation.z * orientation.x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        t4 = +1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = math.atan2(t3, t4)
        return [roll, pitch, yaw]

    def _move_to_start_position(self):
        starting_joint_angles = {'left_w0': 1.58,
                                 'left_w1': 0,
                                 'left_w2': -1.58,
                                 'left_e0': 0,
                                 'left_e1': 0.68,
                                 'left_s0': -0.8,
                                 'left_s1': -0.8}
        self._move_arm(starting_joint_angles)

    def _ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return None
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        else:
            return None
        return limb_joints

    def _move_arm(self, joint_angles):
        self._left_arm.move_to_joint_positions(joint_angles)
        rospy.sleep(1)

    def _move_object_location(self):
        # Generate random y and z locations
        a = random.random() * 2 * math.pi
        r = 0.4 * math.sqrt(random.random())
        y_loc = 0.235 + (r * math.cos(a))
        z_loc = 1.2 + (r * math.sin(a))

        # Create ModelState message
        ball_pose = Pose(position=Point(x=2, y=y_loc, z=z_loc))
        model_state = ModelState(
            model_name='ball', pose=ball_pose, reference_frame="world"
        )

        # Attempt to move ball
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_model_state = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState
            )
            set_model_state(model_state)
            rospy.sleep(1)
            self._obj_x_pos = ball_pose.position.x
            self._obj_y_pos = ball_pose.position.y
            self._obj_z_pos = ball_pose.position.z
        except rospy.ServiceException, e:
            rospy.logerr("Set Model State service call failed: {0}".format(e))

    def _load_gazebo_models(self):
        # Get Models' Path
        model_path = (
            rospkg.RosPack().get_path('retina_reinforcement_sim') + "/models/"
        )

        # Load ball SDF
        ball_xml = ''
        with open(model_path + "simple_ball/model.sdf", "r") as ball_file:
            ball_xml = ball_file.read().replace('\n', '')

        # Spawn ball SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            pose = Pose(position=Point(x=2, y=0.235, z=1.2))
            reference_frame = "world"
            spawn_sdf = rospy.ServiceProxy(
                '/gazebo/spawn_sdf_model', SpawnModel)
            spawn_sdf("ball", ball_xml, "/", pose, reference_frame)
        except rospy.ServiceException, e:
            rospy.logerr("Spawn SDF service call failed: {0}".format(e))

    def _delete_gazebo_models(self):
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model',
                                              DeleteModel)
            delete_model("ball")
        except rospy.ServiceException, e:
            rospy.loginfo("Model service call failed: {0}".format(e))
