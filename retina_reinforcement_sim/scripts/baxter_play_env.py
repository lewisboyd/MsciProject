#!/usr/bin/env python

import sys
import random
import time

import cv2
import rospy

from environment import BaxterEnvironment


def run():
    rospy.init_node("test")
    env = BaxterEnvironment()
    env.reset()
    rospy.on_shutdown(env.close)
    cv2.namedWindow("Feed", cv2.WINDOW_NORMAL)

    # Key action mapping
    actions = {ord('a'): [0.1, 0], ord('s'): [-0.1, 0],
               ord('d'): [0, 0.1], ord('f'): [0, -0.1],
               ord('q'): [0.3, 0], ord('w'): [-0.3, 0],
               ord('e'): [0, 0.3], ord('r'): [0, -0.3]}

    # Run until interupted or 'esc' pressed
    done = True
    key = None
    ep_step = 0
    while not rospy.is_shutdown():
        if done:
            # Reset environment
            state = env.reset()
            print ""
            print "horiz: " + str(state[0])
            print "vert: " + str(state[1])
            print "x pos: " + str(state[2])
            print "y pos: " + str(state[3])
            done = False
            ep_step = 0
        else:
            if key in actions:
                ep_step = ep_step + 1

                # If valid key pressed execute corresponding action
                action = actions[key]
                state, reward, done = env.step(action)
                print ""
                print "action: " + str(action)
                print "horiz: " + str(state[0])
                print "vert: " + str(state[1])
                print "x pos: " + str(state[2])
                print "y pos: " + str(state[3])
                print "reward: " + str(reward)
                print "done: " + str(done)

                done = done or (ep_step == 15)

        # Display image until key pressed
        cv2.imshow("Feed", env.image)
        key = cv2.waitKey(0)

        # If 'esc' pressed exit program
        if key == 27:
            break


if __name__ == '__main__':
    try:
        sys.exit(run())
    except rospy.ROSInterruptException:
        pass
