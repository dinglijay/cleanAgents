  
docker build .   -t defenser:1.0
docker run -ti \
        -p 6028:22 \
        -p 6029:6006 \
        --hostname defenseSer1 \
        --name li-defense2 \
        --gpus 'all' \
        --shm-size 16g \
        --ipc host \
        -v /home/nova:/workspace \
        -v /mnt/DataServer/nova:/DataServer \
        nova-defense