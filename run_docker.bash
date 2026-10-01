#!/bin/bash
# Start (or re-enter) the ROS container with this repo's package mounted in the catkin workspace.
# X access for the container is granted through xhost (no xauth file needed)
xhost local:root

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if docker container inspect ros_container >/dev/null 2>&1; then
    # Container already exists: start it if needed and open a shell in it
    docker start ros_container >/dev/null
    docker exec -it ros_container bash
else
    docker run -it \
           --name=ros_container \
           --env="DISPLAY=$DISPLAY" \
           --env="QT_X11_NO_MITSHM=1" \
           --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
           --volume="$REPO/ros/Attachment-Model:/root/mdk/catkin_ws/src/Attachment-Model" \
           --volume="$REPO/analysis:/root/analysis" \
           --env="NVIDIA_DRIVER_CAPABILITIES=all" \
           --env="NVIDIA_VISIBLE_DEVICES=all" \
           --net=host \
           --privileged \
           --gpus all \
           aung9htet/ubuntu-20.04:miro \
           bash
fi
echo "Done"
