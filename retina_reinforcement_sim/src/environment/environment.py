import image_processor
from camera_controller import CameraController
from time import sleep
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
from geometry_msgs.msg import (
    Pose,
    Point,
)


class Environment:
    """Class to represent the learning environment."""

    def __init__(self):
        baxter_interface.RobotEnable().enable()
        self.left_cam = CameraController()
        self.left_arm = baxter_interface.Limb('left')
        self._load_gazebo_models()

        self.state = None
        self.image = None
        self.prev_dist = None
        self.curr_dist = None
        self.threshold = 50
        self.max_dist = 600

    def reset(self):
        """Start a fresh episode.

        Returns:
            object: New State

        """
        self._move_to_start_position()
        # Due to the gripper the ball may rarely be obscured
        # Loops until ball can be seen (curr_dist is not -1)
        self.curr_dist = -1
        while (self.curr_dist == -1):
            self._move_ball_location()
            self._update_state()
        self.prev_dist = self.threshold
        return self.state

    def step(self, action):
        """Execute the action and update the environment state.

        Returns:
            object: New State
            float: Reward
            boolean: Episode finished

        """
        # If joint move, make move then update state

        self._update_state()
        return self.state, self._reward_joint_move(), (self.curr_dist != -1)

        # If found centre move, if within threshold

    def shutdown(self):
        """Shutdown environment cleanly."""
        self._delete_gazebo_models()

    def _reward_joint_move(self):
        if self.prev_dist < self.threshold:
            return -1
        if self.curr_dist == -1:
            return -1
        return 1 - (self.curr_dist / self.max_dist)

    def _reward_found_move(self):
        if (self.curr_dist < self.threshold):
            return 2
        return -1

    def _update_state(self):
        sleep(1)  # Small wait to ensure sensors up to date
        image_data = self.left_cam.get_image_data()
        self.image, self.state = (
            image_processor.process_image_data(image_data))
        self.prev_dist = self.curr_dist
        self.curr_dist = image_processor.calc_dist(self.image)

    def _move_to_start_position(self):
        starting_joint_angles = {'left_w0': 1.58,
                                 'left_w1': 0,
                                 'left_w2': -1.58,
                                 'left_e0': 0,
                                 'left_e1': 0.68,
                                 'left_s0': -0.8,
                                 'left_s1': -0.8}
        self.left_arm.move_to_joint_positions(starting_joint_angles,
                                              threshold=0.004)

    def _move_ball_location(self):
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
