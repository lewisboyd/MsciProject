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

import rospy
import rospkg
import baxter_interface
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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
from std_msgs.msg import Empty
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


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(95040, 512)
        self.head = nn.Linear(512, 3)

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(1, 95040)
        x = F.relu(self.fc1(x))
        x = self.head(x)
        return x


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(95043, 512)
        self.head = nn.Linear(512, 1)

    def forward(self, image, action):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(1, 95040)
        x = torch.cat((x, action))
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


class ActorCritic():

    def __init__(self):
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.steps_done = 0

        self.actor = Actor()
        self·actor_target = Actor()
        self.critic = Critic()
        self.critic_target = Critic()
        self.memory = ReplayMemory(1000)

    def calc_move(self, cortical_image):
        '''Actor decides a move or a random move is made
        '''

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.actor(cortical_image)
        else:
            return torch.tensor([[random.uniform(0.5, 1.25),
                                  random.uniform(-2, 2),
                                  random.uniform(-2, 2)]], device=self.device,
                                dtype=torch.long)


class Trainer(Thread):
    '''Threaded class to train the DQN

    Args:
        episodes: number of episodes to train for
    '''

    def __init__(self, episodes=100):
        Thread.__init__(self)
        self.episodes = episodes
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.steps_done = 0
        self.memory = ReplayMemory(1000)
        self.prev_cortical_image = np.zeros((246, 468, 3))
        self.curr_cortical_image = np.zeros((246, 468, 3))
        self.curr_dist = 0
        self.prev_dist = 0
        self.threshold = 50
        self.left_arm = baxter_interface.Limb('left')

        print "Starting the CameraListener..."
        self.image_queue = Queue(1)
        self.ready_for_data_event = Event()
        CameraListener(self.image_queue, self.ready_for_data_event)

        print "Creating the ImageProcessor..."
        self.image = np.zeros((1080, 1920, 3))
        self.image_processor = ImageProcessor()

        print "Loading the Networks..."
        self.actor = Actor()
        self.critic = Critic()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.actor.to(self.device)
        self.critic.to(self.device)
        print "Loaded Networks on {}".format(self.device)

    def run(self):
        '''Threads run method to run the training
        '''
        # While still training
        for curr_episode in self.episodes:
            print "Episode {}/{}".format(curr_episode, self.episodes)

            # Position the ball and process initial image data
            self.initialize_episode()

            # Runs until DQN uses found centre_move
            while True:
                # Calculate move to make
                move_values = self.calc_move()

                # Execute the move
                self.execute_move(move_values)

                # Get the wrist image and cortical image
                self.process_image_data()

                # Calculate the pixel distance
                self.curr_dist = self.image_processor.calc_dist(self.image)

                # If move cause sight of ball to be lost then end the episode
                if self.curr_dist == -1:
                    break

                # Evaluate move
                reward = self.evaluate_move(move_values)

                # Save transition in memory
                self.memory.push(self.prev_cortical_image, move_values, reward,
                                 self.curr_cortical_image)

                # If found centre move used then end the episode
                if move_values[0] >= 1:
                    break

                # Update previous pixel distance
                self.prev_dist = self.curr_dist

            # Episode finished when found_centre is used
            self.curr_episode = self.curr_episode + 1

    def initialize_episode(self):
        '''Positions the ball and processes initial image data'''

        self.move_to_start_position()
        move_ball_location()
        sleep(1)  # Wait to ensure camera image is up to date
        self.process_image_data()
        self.prev_dist = self.image_processor.calc_dist(self.image)

        # Ball may be hidden behind gripper so reposition till visible
        if self.prev_dist == -1:
            self.initialize_episode()

    def move_to_start_position(self):
        '''Moves the arms joints back to their starting positions'''

        starting_joint_angles = {'left_w0': 1.58,
                                 'left_w1': 0,
                                 'left_w2': -1.58,
                                 'left_e0': 0,
                                 'left_e1': 0.68,
                                 'left_s0': -0.8,
                                 'left_s1': -0.8}
        self.left_arm.move_to_joint_positions(starting_joint_angles,
                                              threshold=0.004)

    def process_image_data(self):
        '''Gets latest wrist image updating the image attributes
        '''

        self.ready_for_data_event.set()
        image_data = self.image_queue.get()
        self.prev_cortical_image = self.curr_cortical_image
        self.image, self.curr_cortical_image = (
            self.image_processor.process_image_data(image_data)
        )

    def calc_move(self):
        '''Generates a random move or uses the DQN to calculate the move values
        '''

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.dqn(self.curr_cortical_image)
        else:
            return torch.tensor([[random.uniform(0.5, 1.25),
                                  random.uniform(-2, 2),
                                  random.uniform(-2, 2)]], device=self.device,
                                dtype=torch.long)

    def execute_move(self, move_values):
        '''Executes the move according to the given move values'''

        # If found_centre_move used then ignore wrist and elbow move
        if move_values[0] >= 1:
            return

        # Otherwise move the wrist and elbow joints
        wrist_angle = self.left_arm.joint_angle('left_w1') + move_values[1]
        elbow_angle = self.left_arm.joint_angle('left_e1') + move_values[2]
        joint_angles = {'left_w1': wrist_angle,
                        'left_e1': elbow_angle}
        self.left_arm.move_to_joint_positions(
            joint_angles,
            threshold=0.004)

    def evaluate_move(self, move_values):
        '''Evaluates the choosen moves returning a reward'''

        # If found_centre moved return big reward if within distance threshold
        if move_values[0] >= 1:
            if self.curr_dist < self.threshold:
                return 3
        # If closed distance to ball return a small reward
        if self.curr_dist < self.prev_dist:
            return 1
        # Otherwise return a negative reward
        return -1


class ImageProcessor:
    '''Class to processes the ROS image data

    Attributes:
        retina (object): CUDA enhanced retina
        cortex (object): CUDA enhanced cortex
        bridge (object): CvBridge to convert ROS image data to OpenCV image
    '''

    def __init__(self):
        self.retina, self.cortex = self._create_retina_and_cortex()
        self.bridge = CvBridge()

    def process_image_data(self, image_data):
        '''Creates the cortical image and calculates the distance of the ball
        to the centre of the image

        Args:
            image_data (object): The ROS image message

        Returns:
            object: The cortical image
            float: The distance of the ball to the centre of the image
        '''
        # Convert to OpenCV image
        image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")

        # Sample with retina
        cortical_image = self._retina_sample(image)

        return image, cortical_image,

    def calc_dist(self, image):
        ''' Calculates distance from image centre to ball centre

        Args:
            image: image to find ball in

        Returns:
            distance of ball to centre of image, or -1 if unsuccessful
        '''
        moments = cv2.moments(self._create_mask(image))
        if (moments["m00"] == 0):
            return -1
        x_centre = int(moments["m10"] / moments["m00"])
        y_centre = int(moments["m01"] / moments["m00"])
        x_dist = 960 - x_centre
        y_dist = 540 - y_centre
        return int(math.sqrt((x_dist ** 2) + (y_dist ** 2)))

    def _retina_sample(self, image):
        '''Samples with the retina returning the cortical image

        Returns:
            The cortical image
        '''
        v_c = self.retina.sample(image)
        l_c = self.cortex.cort_image_left(v_c)
        r_c = self.cortex.cort_image_right(v_c)
        return np.concatenate((np.rot90(l_c), np.rot90(r_c, k=3)), axis=1)

    def _create_mask(self, image):
        '''Processes the image to obtain a binary mask showing the ball

        Args:
            image: BGR OpenCV image

        Returns:
            Binary mask
        '''
        # Define lower and upper colour bounds
        ORANGE_MIN = np.array([5, 50, 50], np.uint8)
        ORANGE_MAX = np.array([15, 255, 255], np.uint8)

        # Threshold the image in the HSV colour space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, ORANGE_MIN, ORANGE_MAX)

        # Return the mask
        return mask

    def _create_retina_and_cortex(self):
        '''Creates the retina and cortex

        Returns:
            The CUDA enhanced Retina and Cortex
        '''
        # Load in data
        retina_path = '/home/lewis/Downloads/RetinaCUDA-master/Retinas'
        with open(retina_path + '/ret50k_loc.pkl', 'rb') as handle:
            loc50k = pickle.load(handle)
        with open(retina_path + '/ret50k_coeff.pkl', 'rb') as handle:
            coeff50k = pickle.load(handle)

        # Create retina and cortex
        L, R = cortex.LRsplit(loc50k)
        L_loc, R_loc = cortex.cort_map(L, R)
        L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)
        ret = retina_cuda.create_retina(loc50k, coeff50k,
                                        (1080, 1920, 3), (960, 540))
        cort = cortex_cuda.create_cortex_from_fields_and_locs(
            L, R, L_loc, R_loc, cort_size, gauss100=G, rgb=True
        )

        # Return the retina and cortex
        return ret, cort


class CameraListener:
    '''Class to handle ROS image data

    Attributes:
        image_queue (object): Thread safe queue used to send data when
            requested
        ready_for_data_event (object): Event to signal if image_queue needs to
            be populated
        sub (object): Subscription to Baxter's left arm camera topic
    '''

    def __init__(self, image_queue, ready_for_data_event):
        self.image_queue = image_queue
        self.ready_for_data_event = ready_for_data_event
        self.sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image,
                                    self.callback, queue_size=1,
                                    buff_size=20000000)

    def callback(self, image_data):
        '''Callback function to populate image_queue when ready_for_data_event
        is set

        Args:
            image_data: ROS image data
        '''
        if self.ready_for_data_event.is_set():
            self.image_queue.put(image_data)
            self.ready_for_data_event.clear()
        else:
            return


def move_ball_location():
    '''Moves the ball to a random position within the retinas field of view
    '''
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


def load_gazebo_models():
    '''Loads in the models via the Spawning service
    '''
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
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_sdf("ball", ball_xml, "/", pose, reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))


def delete_gazebo_models():
    '''Deletes the models via the Deletion service
    '''
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model("ball")
    except rospy.ServiceException, e:
        rospy.loginfo("Model service call failed: {0}".format(e))


def shutdown():
    '''Called when the node is shutdown to delete the models and OpenCV windows
    '''
    delete_gazebo_models()
    cv2.destroyAllWindows()


def main():
    print "Loading world..."
    rospy.init_node("learning_env")
    rospy.on_shutdown(shutdown)
    load_gazebo_models()
    rospy.wait_for_message("/robot/sim/started", Empty)

    print "Enabling Baxter..."
    rs = baxter_interface.RobotEnable()
    rs.enable()

    print "Creating the trainer..."
    trainer = Trainer()
    trainer.setDaemon(True)

    print "Training started, press ctrl-c to exit..."
    trainer.start()
    rospy.spin()


if __name__ == '__main__':
    sys.exit(main())
