import sys
import random

import cv2
import rospy

from environment import BaxterEnvironment


def run():
    rospy.init_node("test")
    env = BaxterEnvironment()
    rospy.on_shutdown(env.close)

    cv2.namedWindow("Feed", cv2.WINDOW_NORMAL)
    env.reset()

    while not rospy.is_shutdown():
        action = [random.uniform(-1, 1), random.uniform(-1, 1)]
        state, reward, done = env.step(action)

        print ""
        print "horiz: " + str(state[0])
        print "vert: " + str(state[1])
        print "reward: " + str(reward)
        print "done: " + str(done)

        img = env.image
        cv2.imshow("Feed", img)
        cv2.waitKey(5)


if __name__ == '__main__':
    try:
        sys.exit(run())
    except rospy.ROSInterruptException:
        pass
