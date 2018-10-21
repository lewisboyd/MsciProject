#!/usr/bin/env python

import sys
import math
from threading import Thread
from threading import Event
from Queue import Queue

import rospy
import rospkg
import baxter_interface
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import scipy
import cPickle as pickle
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

sys.path.append('/home/lewis/Downloads/RetinaCUDA-master/py')
sys.path.append(
    '/home/lewis/Downloads/RetinaCUDA-master/py/Piotr_Ozimek_retina')
import retina_cuda
import cortex_cuda
import retina
import cortex

class Trainer(Thread):


    def __init__(self, episodes=100):
        Thread.__init__(self)
        self.curr_episode = 0
        self.episodes = episodes
        self.cortical_image = np.zeros((246, 468, 3))
        self.curr_dist_from_centre = 0
        self.prev_dist_from_centre = 0

        print "Starting the CameraListener..."
        self.image_queue = Queue(1)
        self.ready_for_data_event = Event()
        CameraListener(self.image_queue, self.ready_for_data_event)

        print "Creating the ImageProcessor..."
        ret, cort = create_retina_and_cortex()
        self.image_processor = ImageProcessor(ret, cort)

    def run(self):
        # While still training
        while self.curr_episode < self.episodes:
            print "Episode {}/{}".format(self.curr_episode, self.episodes)

            # Move ball location
            #

            # Move arm back to start position
            move_to_start_position()

            # Populate initial cortical_image and prev_dist_from_centre
            self.process_initial_image()

            print "distance {}".format(self.prev_dist_from_centre)
            cv2.namedWindow("feed", cv2.WINDOW_NORMAL)
            cv2.imshow("feed", self.cortical_image)
            cv2.waitKey(1000)

            # Runs until DQN uses found centre_move
            i = 0
            while True:
                # DQN makes move using cortical_image input
                i = i + 1

                starting_joint_angles = {'left_w0': 1.58,
                                         'left_w1': 0.1,
                                         'left_w2': -1.58,
                                         'left_e0': 0,
                                         'left_e1': 0.68,
                                         'left_s0': -0.8,
                                         'left_s1': -0.8}
                baxter_interface.Limb('left').move_to_joint_positions(
                    starting_joint_angles,
                    threshold=0.004)

                # Update cortical_image and get curr_dist_from_centre
                self.process_image()
                print "distance {}".format(self.curr_dist_from_centre)
                cv2.imshow("feed", self.cortical_image)
                cv2.waitKey(1000)

                # Evaluate move by DQN, if found centre move used then break
                if (i == 3):
                    break

            # Episode finished when found_centre is used
            self.curr_episode = self.curr_episode + 1

    def process_initial_image(self):
        self.ready_for_data_event.set()
        image_data = self.image_queue.get()
        self.cortical_image, self.prev_dist_from_centre = self.image_processor.process_image(image_data)

    def process_image(self):
        self.ready_for_data_event.set()
        image_data = self.image_queue.get()
        self.cortical_image, self.curr_dist_from_centre = self.image_processor.process_image(image_data)



class ImageProcessor:
    '''Class to processes the ROS image data

    Attributes:
        retina (object): CUDA enhanced retina
        cortex (object): CUDA enhanced
        bridge (object): CvBridge to convert ROS image data to OpenCV image
    '''

    def __init__(self, ret, cort):
        self.retina = ret
        self.cortex = cort
        self.bridge = CvBridge()
        self.image = np.zeros((1080, 1920, 3))
        self.cortical_image = np.zeros((246, 468, 3))
        self.image_mask = np.zeros((1080, 1920))
        self.dist_from_centre = 0

    def process_image(self, image_data):
        '''Creates the cortical image and calculates the distance of the ball
        to the centre of the image

        Args:
            image_data (object): The ROS image message

        Returns:
            object: The cortical image
            float: The distance of the ball to the centre of the image
        '''
        # Convert to OpenCV image
        self.image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")

        # Sample with retina
        self.cortical_image = self.retina_sample()

        # Obtain mask of the image
        self.image_mask = self.create_mask(self.image)

        # Calculate distance of ball from centre of camera
        self.calc_dist_from_centre()

        print "ImageProcessor distance {}".format(self.dist_from_centre)
        return self.cortical_image, self.dist_from_centre

    def retina_sample(self):
        '''Samples with the retina returning the cortical image

        Returns:
            object: The cortical image
        '''
        v_c = self.retina.sample(self.image)
        l_c = self.cortex.cort_image_left(v_c)
        r_c = self.cortex.cort_image_right(v_c)
        c_c = np.concatenate((np.rot90(l_c),np.rot90(r_c,k=3)),axis=1)
        return c_c

    def create_mask(self, image):
        '''Processes the image to obtain a binary mask showing the ball

        Args:
            image (object): BGR OpenCV image

        Returns:
            object: Binary mask
        '''
        # Define lower and upper colour bounds
        ORANGE_MIN = np.array([5, 50, 50],np.uint8)
        ORANGE_MAX = np.array([15, 255, 255],np.uint8)

        # Threshold the image in the HSV colour space
        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, ORANGE_MIN, ORANGE_MAX)

        # Return the mask
        return mask

    def calc_dist_from_centre(self):
        ''' Calculates the distance from the centre of the image to the centre
        of the ball
        '''
        moments = cv2.moments(self.image_mask)
        x_centre = int(moments["m10"] / moments["m00"])
        y_centre = int(moments["m01"] / moments["m00"])
        x_dist = 960 - x_centre
        y_dist = 540 - y_centre
        self.dist_from_centre = math.sqrt((x_dist ** 2) + (y_dist ** 2))

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
            image_data (object): ROS image data
        '''
        if self.ready_for_data_event.is_set():
            self.image_queue.put(image_data)
            self.ready_for_data_event.clear()
        else:
            return


def create_retina_and_cortex():
    '''Creates the retina and cortex

    Returns:
        object: The CUDA enhanced Retina
        object: The CUDA enhanced Cortex
    '''
    # Load in data
    retina_path = '/home/lewis/Downloads/RetinaCUDA-master/Retinas'
    mat_data = '/home/lewis/Downloads/RetinaCUDA-master/Retinas'
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
    cort = cortex_cuda.create_cortex_from_fields_and_locs(L, R, L_loc,
        R_loc, cort_size, gauss100=G, rgb=True)

    # Return the retina and cortex
    return ret, cort


def load_gazebo_models():
    '''Loads in the models via the Spawning service
    '''
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('retina_reinforcement_sim')+"/models/"

    # Load ball SDF
    ball_xml = ''
    with open (model_path + "simple_ball/model.sdf", "r") as ball_file:
        ball_xml=ball_file.read().replace('\n', '')

    # Spawn ball SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        pose = Pose(position=Point(x=2, y=0.235, z=1.186))
        reference_frame = "world"
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf = spawn_sdf("ball", ball_xml, "/",
                             pose, reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))


def delete_gazebo_models():
    '''Deletes the models via the Deletion service
    '''
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("ball")
    except rospy.ServiceException, e:
        rospy.loginfo("Model service call failed: {0}".format(e))


def shutdown():
    '''Called when the node is shutdown to delete the models and OpenCV windows
    '''
    delete_gazebo_models()
    cv2.destroyAllWindows()


def move_to_start_position():
    '''Moves the arms joints back to their starting positions
    '''
    starting_joint_angles = {'left_w0': 1.58,
                             'left_w1': 0,
                             'left_w2': -1.58,
                             'left_e0': 0,
                             'left_e1': 0.68,
                             'left_s0': -0.8,
                             'left_s1': -0.8}
    baxter_interface.Limb('left').move_to_joint_positions(
        starting_joint_angles,
        threshold=0.004)


def main():
    print "Loading world..."
    rospy.init_node("learning_env")
    rospy.on_shutdown(shutdown)
    load_gazebo_models()
    rospy.wait_for_message("/robot/sim/started", Empty)

    print "Enabling Baxter..."
    rs = baxter_interface.RobotEnable()
    rs.enable()

    print "Moving to start pose..."
    move_to_start_position()

    print "Creating the trainer"
    trainer = Trainer()
    trainer.daemon = True

    print "Starting the trainer, press ctrl-c to exit..."
    trainer.start()
    rospy.spin()


if __name__ == '__main__':
    sys.exit(main())
