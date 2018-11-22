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
    """Class representing the learning environment."""

    def __init__(self):
        """Load in environment and initialise state."""
        baxter_interface.RobotEnable().enable()
        self._load_gazebo_models()

        self.state = None

        self._left_cam = CameraController()
        self._left_arm = baxter_interface.Limb('left')
        self._max_dist = 600.0
        self._threshold = 30
        self._curr_dist = None
        self._prev_dist = self._threshold
        self._image = None
        self._WRIST_LIMIT = 0.35
        self._ELBOW_LIMIT = 0.26

    def reset(self):
        """Start a fresh episode.

        Moves the ball to a random visible location and resets the
        environment's state.

        Returns:
            object: New State

        """
        self._move_to_start_position()
        # Due to the gripper the ball may rarely be obscured
        # Loops until ball can be seen (distance isn't -1)
        visible = False
        while not visible:
            self._move_ball_location()
            self._update_state()
            visible = image_processor.calc_dist(self.state) != -1
        self._prev_dist = self._threshold
        return self.state

    def step(self, found_value, wrist_value, elbow_value):
        """Execute the action.

        If found_action is true then if the centre of the screen is within the
        threshold distance to the ball's centre then the reward is 2 and the
        episode is finished.

        If found_action is false then moves the baxter robot's left arm wrist
        and elbow by the given values.
        If sight of the ball is lost then the reward is -1 and the episode is
        finished.
        If the distance to the ball's centre was less than the threshold value
        before the move then the reward is -1.
        Otherwise returns a positive reward scaling from 0 to 1 with the
        distance to the ball's centre.

        Args:
            found_value (float) : If > 0 then executes found action
            wrist_value (float) : If not found_action then moves wrist
            elbow_value (float) : If not found_action then moves elbow

        Return
            object : New State
            float : Reward
            bool : Episode finished

        """
        found_action, wrist_action, elbow_action = (
            self._parse_action_values(found_value, wrist_value, elbow_value))
        if found_action:
            return self._evaluate_found_action()
        else:
            self._move_arm(wrist_action, elbow_action)
            self._update_state()
            return self._evaluate_move_action()

    def shutdown(self):
        """Delete loaded models. To be called when node is shutting down."""
        self._delete_gazebo_models()

    def _parse_action_values(self, found_value, wrist_value, elbow_value):
        found_move = True if found_value > 0 else False
        wrist_move = wrist_value * self._WRIST_LIMIT
        elbow_move = elbow_value * self._ELBOW_LIMIT
        return found_move, wrist_move, elbow_move

    def _evaluate_found_action(self):
        if self._curr_dist < self._threshold:
            return self.state, 2, True
        else:
            return self.state, -1, False

    def _evaluate_move_action(self):
        visible = image_processor.calc_dist(self.state) != -1
        if not visible:
            return self.state, -1, True
        elif self._prev_dist < self._threshold:
            return self.state, -1, False
        else:
            return self.state, 1.0 - (self._curr_dist / self._max_dist), False

    def _update_state(self):
        sleep(1)  # Small wait to ensure sensors up to date
        image_data = self._left_cam.get_image_data()
        self._image, self.state = (
            image_processor.process_image_data(image_data))
        self._prev_dist = self._curr_dist
        self._curr_dist = image_processor.calc_dist(self._image)

    def _move_to_start_position(self):
        starting_joint_angles = {'left_w0': 1.58,
                                 'left_w1': 0,
                                 'left_w2': -1.58,
                                 'left_e0': 0,
                                 'left_e1': 0.68,
                                 'left_s0': -0.8,
                                 'left_s1': -0.8}
        self._left_arm.move_to_joint_positions(starting_joint_angles,
                                               threshold=0.004)

    def _move_arm(self, wrist, elbow):
        wrist_angle = self._left_arm.joint_angle('left_w1') + wrist
        elbow_angle = self._left_arm.joint_angle('left_e1') + elbow
        joint_angles = {'left_w1': wrist_angle,
                        'left_e1': elbow_angle}
        self._left_arm.move_to_joint_positions(
            joint_angles, threshold=0.004)

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
