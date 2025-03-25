FROM osrf/ros:noetic-desktop-full

# Set up environment
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies and catkin tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    python3-catkin-tools \
    tmux \
    python3-opencv \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter Notebook and other Python dependencies
RUN pip3 install jupyter notebook numpy opencv-python matplotlib ipywidgets

# Create a catkin workspace
WORKDIR /ros_ws
RUN mkdir -p /ros_ws/src

# Initialize the workspace with catkin tools
RUN source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin config --extend /opt/ros/noetic && \
    catkin build

# Add source command to bashrc
RUN echo "source /ros_ws/devel/setup.bash" >> ~/.bashrc

# Add basic tmux configuration
RUN echo "set -g mouse on" > ~/.tmux.conf && \
    echo "set -g history-limit 10000" >> ~/.tmux.conf

# Set up entrypoint
COPY ./ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

# Exposed port for Jupyter Notebook
EXPOSE 8888

CMD [ "jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]