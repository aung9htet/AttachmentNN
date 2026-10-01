cmake_minimum_required(VERSION 3.0.2)
project(attachment_model)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Action.msg
  Emotion.msg
  RobotPub.msg
  Perception.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  rospy
  std_msgs
  message_runtime
)

## Scripts will be run directly from src/ directory via roslaunch
## No need to install them with catkin_install_python
