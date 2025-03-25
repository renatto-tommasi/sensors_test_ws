#!/bin/bash
set -e

# Setup ROS environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
if [ -f /ros_ws/devel/setup.bash ]; then
  source /ros_ws/devel/setup.bash
fi

# Execute the command passed to this script
exec "$@"