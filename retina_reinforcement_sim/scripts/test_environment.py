#!/usr/bin/env python

from environment import Environment, image_processor
import rospy
import cv2


if __name__ == '__main__':
    rospy.init_node("test_environment")

    env = Environment()
    cam_controller = env._left_cam

    rospy.on_shutdown(env.shutdown)

    env._move_to_start_position()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("cortical image", cv2.WINDOW_NORMAL)

    while not rospy.is_shutdown():
        img_data = cam_controller.get_image_data()
        img, cort_img = image_processor.process_image_data(img_data)

        dist = image_processor.calc_dist(img)
        visible = image_processor.calc_dist(cort_img) != -1

        print "Distance : " + str(dist)
        print "Visible : " + str(visible)

        cv2.imshow("image", img)
        cv2.imshow("cortical image", cort_img)

        cv2.waitKey(50)
