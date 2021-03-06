#!/bin/bash
# 
# 启动, 镜像：mxnet-cu90-opencv-py3   gpu_cu101
# Author: alex
# Created Time: 2019年03月26日 星期二 15时13分07秒
docker run --rm -ti --runtime=nvidia --name insightface \
    -p 20920:20920 \
    -v /var/www/face_models:/models \
    -v /var/www/tmp/faces:/var/www/tmp/faces \
    -v `pwd`:/faces \
    -e MXNET_CUDNN_AUTOTUNE_DEFAULT=0 \
    -e PYTHONIOENCODING=utf-8 \
    -w /faces \
    "$1" /bin/bash
