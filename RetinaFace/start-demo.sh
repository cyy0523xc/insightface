#!/bin/bash
# 
# 启动
# ./start.sh registry.cn-hangzhou.aliyuncs.com/ibbd/face:gpu_cu101 \
#     python3 demo.py
# Author: alex
# Created Time: 2019年03月26日 星期二 15时13分07秒
cmd=$*
if [ $# -le 2 ]; then
    cmd="$* /bin/bash"
fi
echo "Command: $cmd"

docker rm -f ibbd-face
docker run --rm -d --runtime=nvidia --name ibbd-face \
    -p 20920:20920 \
    -v `pwd`:/faces \
    -v `pwd`/../tests:/faces/tests \
    -v /var/www/face_models:/models \
    -e MXNET_CUDNN_AUTOTUNE_DEFAULT=0 \
    -e PYTHONIOENCODING=utf-8 \
    -w /faces \
    $cmd
