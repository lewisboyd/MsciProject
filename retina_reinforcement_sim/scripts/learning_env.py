#!/usr/bin/env python

import sys

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


class Listener:
    '''Class to process the image data from Baxter's left hand wrist camera

    Attributes:
        retina (object): CUDA enhanced retina
        cortex (object): CUDA enhanced cortex
        bridge (object): CvBridge to convert ROS image data to OpenCV image
        sub (object): Subscription to Baxter's left arm camera topic
    '''

    def __init__(self):
        self.retina, self.cortex = self.create_retina_and_cortex()
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("/cameras/left_hand_camera/image", Image,
                                    self.callback, queue_size=1,
                                    buff_size=2**25)

    def callback(self, data):
        '''Callback function to process the ROS image data

        Args:
            data (object): ROS image data
        '''
        try:
            # Convert to cv image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Sample with retina
            cortical_image = self.sample(cv_image)

            # Display the cortical image
            cv2.namedWindow("Retina Feed", cv2.WINDOW_NORMAL)
            cv2.imshow("Retina Feed", cortical_image)

            # Small wait else image will not display
            cv2.waitKey(1)
        except Exception as e:
            # Print error
            print e.message

    def sample(self, image):
        '''Samples with the retina returning the cortical image

        Args:
            image (object): BGR OpenCV image to be sampled

        Returns:
            object: The cortical image
        '''
        v_c = self.retina.sample(image)
        l_c = self.cortex.cort_image_left(v_c)
        r_c = self.cortex.cort_image_right(v_c)
        c_c = np.concatenate((np.rot90(l_c),np.rot90(r_c,k=3)),axis=1)
        return c_c


    def create_retina_and_cortex(self):
        '''Creates the retina and cortex used by the Listener

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
        # ret = retina_cuda.create_retina(loc50k, coeff50k,
        #     (800, 800, 3), (400, 400))
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
        pose = Pose(position=Point(x=5.0, y=0.0, z=2.0))
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


def main():
    rospy.init_node("learning_env")

    # Load Gazebo Models via Spawning Services
    load_gazebo_models()

    # Delete models and close OpenCV windows on shutdown
    rospy.on_shutdown(shutdown)

    # Wait for the All Clear from emulator startup
    rospy.wait_for_message("/robot/sim/started", Empty)

    print "Enabling Baxter..."
    rs = baxter_interface.RobotEnable()
    rs.enable()

    print "Starting listener..."
    listener = Listener()

    rospy.spin()


if __name__ == '__main__':
    sys.exit(main())
