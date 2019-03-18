import sys
import random
import timeit

import cv2
import rospy

from environment import BaxterEnvironment


def run_random(waitTime=0, verbose=True):
    rospy.init_node("test")
    env = BaxterEnvironment()
    env.reset()
    rospy.on_shutdown(env.close)
    cv2.namedWindow("Feed", cv2.WINDOW_NORMAL)

    # Run randomly untill interupted
    done = True:
    while not rospy.is_shutdown():
        if done:
            # Reset
            state = env.reset()
            if verbose:
                print ""
                print "horiz: " + str(state[0])
                print "vert: " + str(state[1])
            done = False
        else:
            # Random action
            action = [random.uniform(-1, 1), random.uniform(-1, 1)]
            state, reward, done = env.step(action)
            if verbose:
                print ""
                print "action: " + str(action)
                print "horiz: " + str(state[0])
                print "vert: " + str(state[1])
                print "reward: " + str(reward)
                print "done: " + str(done)
        # Display camera image
        cv2.imshow("Feed", env.img)
        cv2.waitKey(waitTime)


if __name__ == '__main__':
    try:
        sys.exit(run_random())
    except rospy.ROSInterruptException:
        pass
