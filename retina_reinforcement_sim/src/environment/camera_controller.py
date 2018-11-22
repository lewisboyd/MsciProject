import rospy
from threading import Event
from Queue import Queue
from sensor_msgs.msg import Image


class CameraController:
    """Class to get most recent image from a camera's topic."""

    def __init__(self, image_topic="/cameras/left_hand_camera/image"):
        """Start a subcriber to populate the queue with an image when called.

        Args:
            image_topic (str): topic for the camera's image messages

        """
        self.image_queue = Queue(1)
        self.ready_for_data_event = Event()
        rospy.Subscriber(image_topic, Image, self._callback, queue_size=1,
                         buff_size=20000000)

    def _callback(self, image_data):
        if self.ready_for_data_event.is_set():
            self.image_queue.put(image_data)
            self.ready_for_data_event.clear()

    def get_image_data(self):
        """Return the next image message."""
        self.ready_for_data_event.set()
        return self.image_queue.get()
