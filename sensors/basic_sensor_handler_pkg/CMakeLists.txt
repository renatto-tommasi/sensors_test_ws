cmake_minimum_required(VERSION 3.0.2)
project(basic_sensor_handler_pkg)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
  geometry_msgs
  cv_bridge
  tf2_ros
  tf2_geometry_msgs
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    visualization_msgs
    geometry_msgs
    cv_bridge
    tf2_ros
    tf2_geometry_msgs
)

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/3d_lidar.py
  scripts/rgbd_camera.py
  scripts/fisheye_cameras.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
