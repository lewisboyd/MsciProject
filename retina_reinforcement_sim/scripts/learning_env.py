#!/usr/bin/env python

import sys
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


class ImageProcessor(Thread):
    '''Threaded class to processes the ROS image data

    Attributes:
        retina (object): CUDA enhanced retina
        cortex (object): CUDA enhanced cortex
        image_queue (object): Thread safe queue shared with CameraListener
        processing_event (object): Event used to signal when need more data
        bridge (object): CvBridge to convert ROS image data to OpenCV image
    '''

    def __init__(self, ret, cort, image_queue, processing_event):
        Thread.__init__(self)
        self.retina = ret
        self.cortex = cort
        self.image_queue = image_queue
        self.processing_event = processing_event
        self.bridge = CvBridge()

    def run(self):
        '''Processes the image data when the processing_event is set to true
        '''
        while True:
            self.processing_event.wait()
            image_data = self.image_queue.get()
            self.process_image(image_data)
            self.processing_event.clear()

    def process_image(self, image_data):
        try:
            print "Processing image"
            # Convert to cv image
            cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")

            # Sample with retina
            cortical_image = self.retina_sample(cv_image)

            # Display the cortical image
            cv2.namedWindow("Retina Feed", cv2.WINDOW_NORMAL)
            cv2.imshow("Retina Feed", cortical_image)

            # Obtain mask of the image
            mask = self.create_mask(cv_image)

            # Display the mask
            cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
            cv2.imshow("Mask", mask)

            # Obtain mask of the cortical image
            mask = self.create_mask(cortical_image)

            # Display the mask
            cv2.namedWindow("Cortical Mask", cv2.WINDOW_NORMAL)
            cv2.imshow("Cortical Mask", mask)

            # Small wait else images will not display
            print "Waiting"
            cv2.waitKey(5000)

            if self.start:
                print "Moving to start"
                move_to_start_position()
                self.start = False
            else:
                print "Moving"
                starting_joint_angles = {'left_w0': 1.58,
                                         'left_w1': 1,
                                         'left_w2': -1.58,
                                         'left_e0': 0,
                                         'left_e1': 0.7,
                                         'left_s0': -0.8,
                                         'left_s1': -0.8}
                baxter_interface.Limb('left').move_to_joint_positions(
                    starting_joint_angles,
                    threshold=0.004)
                self.start = True
            print "Finished move"
        except Exception as e:
            # Print error
            print e.message

    def retina_sample(self, image):
        '''Samples with the retina returning the cortical image

        Args:
            image (object): BGR OpenCV image

        Returns:
            object: The cortical image
        '''
        v_c = self.retina.sample(image)
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


class Listener:
    '''Class to handle ROS image data

    Attributes:
        image_queue (object): Thread safe queue shared with ImageProcessor
        processing_event (object): Event used to signal if ImageProcessor is
            processing
        sub (object): Subscription to Baxter's left arm camera topic
    '''

    def __init__(self, image_queue, processing_event):
        self.image_queue = image_queue
        self.processing_event = processing_event
        self.sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image,
                                    self.callback, queue_size=1,
                                    buff_size=20000000)

    def callback(self, image_data):
        '''Callback function to give ImageProcessor most up to date image data

        Args:
            image_data (object): ROS image data
        '''
        if self.processing_event.is_set():
            return
        else:
            self.image_queue.put(image_data)
            self.processing_event.set()


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
                             'left_e1': 0.7,
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

    print "Starting the ImageProcessor..."
    image_queue = Queue(1)
    processing_event = Event()
    ret, cort = create_retina_and_cortex()
    image_processor = ImageProcessor(ret, cort, image_queue, processing_event)
    image_processor.daemon = True
    image_processor.start()

    print "Starting the CameraListener..."
    Listener(image_queue, processing_event)

    print "Running, press ctrl-c to exit..."
    rospy.spin()


if __name__ == '__main__':
    sys.exit(main())
