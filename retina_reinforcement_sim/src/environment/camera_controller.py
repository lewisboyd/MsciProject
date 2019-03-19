from Queue import Queue
from threading import Event

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraController:
    """Class to get the most recent image from a camera's topic."""

    def __init__(self, img_topic, img_size):
        """Start a subcriber to populate the queue with an image when called.

        Args:
            img_topic (str): topic for the camera's image messages
            img_size (tuple): size (height, width) of image to return

        """
        self.img_queue = Queue(1)
        self.ready_for_data_event = Event()
        self.img_size = img_size
        self.bridge = CvBridge()
        self.time = rospy.Time.now()
        rospy.Subscriber(img_topic, Image, self._callback, queue_size=1,
                         buff_size=20000000)
        # buff_size = 800 * 1280 * 3 * 1.25
        # rospy.Subscriber(img_topic, Image, self._callback, queue_size=1,
        #                  buff_size=buff_size)

    def _callback(self, image_data):
        if self.ready_for_data_event.is_set():
            # if image_data.header.stamp - self.time >= 0:
            #     self.img_queue.put(image_data)
            #     self.ready_for_data_event.clear()
            self.img_queue.put(image_data)
            self.ready_for_data_event.clear()

    def get_image(self):
        """Get and preprocess the most recent image."""
        rospy.sleep(0.2)
        # self.time = rospy.Time.now()
        self.ready_for_data_event.set()
        img_data = self.img_queue.get()
        img = self.bridge.imgmsg_to_cv2(img_data, "rgb8")
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        return img
