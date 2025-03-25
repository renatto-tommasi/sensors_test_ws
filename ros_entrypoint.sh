#!/bin/bash
set -e

# Setup ROS environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
if [ -f /ros_ws/devel/setup.bash ]; then
    source /ros_ws/devel/setup.bash
fi

# Start Jupyter Notebook in the background
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &

# Execute the command passed to docker run
exec "$@"