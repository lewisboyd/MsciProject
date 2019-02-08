cmake_minimum_required(VERSION 2.8.3)
project(retina_reinforcement_sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  rospack
  baxter_core_msgs
  baxter_gazebo
  baxter_interface
  baxter_tools
  gazebo_ros
  gazebo_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
  rospy
  rospack
  baxter_core_msgs
  baxter_gazebo
  baxter_interface
  baxter_tools
  gazebo_ros
  gazebo_msgs
)

#############
## Install ##
#############

install(PROGRAMS
   scripts/train_ddpg.py
   scripts/test_environment.py
   scripts/pendulum.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

foreach(dir launch models)
   install(DIRECTORY ${dir}/
      DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
