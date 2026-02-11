export CONTAINER_ID=`docker ps -lq`
docker commit $CONTAINER_ID aung9htet/ubuntu-20.04:miro
echo "done"
