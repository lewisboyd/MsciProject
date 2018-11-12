import rospy
from threading import Event
from Queue import Queue
from sensor_msgs.msg import Image


class CameraController:
    '''Class to start camera image subscriber and provide method to get most
    recent ROS image message
    '''

    def __init__(self):
        self.image_queue = Queue(1)
        self.ready_for_data_event = Event()
        rospy.Subscriber("/cameras/left_hand_camera/image", Image,
                         self._callback, queue_size=1, buff_size=20000000)

    def _callback(self, image_data):
        '''Callback function to populate image_queue when required

        Args:
            image_data: ROS image data
        '''
        if self.ready_for_data_event.is_set():
            self.image_queue.put(image_data)
            self.ready_for_data_event.clear()

    def get_image_data(self):
        '''Returns the most recent image
        '''
        self.ready_for_data_event.set()
        return self.image_queue.get()
