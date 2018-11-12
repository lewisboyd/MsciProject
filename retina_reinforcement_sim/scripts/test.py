#!/usr/bin/env python

from environment import Environment
import sys
import rospy
import cv2


def main():
    print "Loading environment..."
    env = Environment()
    env.reset()

    def shutdown():
        '''Called when the node is shutdown to delete the models and OpenCV
        windows
        '''
        env.shutdown()
        cv2.destroyAllWindows()

    rospy.on_shutdown(shutdown)

    image = env.image
    cv2.namedWindow("Image 1", cv2.WINDOW_NORMAL)
    cv2.imshow("Image 1", image)
    cv2.waitKey(500)

    env.reset()

    image = env.image
    cv2.namedWindow("Image 2", cv2.WINDOW_NORMAL)
    cv2.imshow("Image 2", image)
    cv2.waitKey(500)

    env.reset()

    image = env.image
    cv2.namedWindow("Image 3", cv2.WINDOW_NORMAL)
    cv2.imshow("Image 3", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node("learning_env")
    sys.exit(main())
