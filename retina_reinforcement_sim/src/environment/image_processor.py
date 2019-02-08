import cortex
import cortex_cuda
import retina_cuda
import math

from cv_bridge import CvBridge
import cv2
import numpy as np
import cPickle as pickle


def _create_retina_and_cortex():
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


ret, cort = _create_retina_and_cortex()
bridge = CvBridge()


def process_image_data(image_data):
    """Create an OpenCV RGB image and cortical image from the image data.

    Args:
        image_data (object): ROS image message

    Returns:
        object: OpenCV RGB image
        object: Cortical RGB image

    """
    global bridge

    # Convert to OpenCV image
    image = bridge.imgmsg_to_cv2(image_data, "rgb8")

    # Sample with retina
    cortical_image = _retina_sample(image)

    return image, cortical_image


def calc_dist(image):
    """Calculate the distance from the image's centre to ball's centre.

    Args:
        image: RGB OpenCV image

    Returns:
        distance of ball to centre of image, or -1 if unsuccessful

    """
    moments = cv2.moments(_create_mask(image))
    if (moments["m00"] == 0):
        return -1
    x_centre = int(moments["m10"] / moments["m00"])
    y_centre = int(moments["m01"] / moments["m00"])
    x_dist = 960 - x_centre
    y_dist = 540 - y_centre
    return int(math.sqrt((x_dist ** 2) + (y_dist ** 2)))


def _retina_sample(image):
    global ret
    global cort

    v_c = ret.sample(image)
    l_c = cort.cort_image_left(v_c)
    r_c = cort.cort_image_right(v_c)
    return np.concatenate((np.rot90(l_c), np.rot90(r_c, k=3)), axis=1)


def _create_mask(image):
    # Define lower and upper colour bounds
    ORANGE_MIN = np.array([5, 50, 50], np.uint8)
    ORANGE_MAX = np.array([15, 255, 255], np.uint8)

    # Threshold the image in the HSV colour space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, ORANGE_MIN, ORANGE_MAX)

    # Return the mask
    return mask
