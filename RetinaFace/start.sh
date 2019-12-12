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

docker rm -f insightface
docker run -d --runtime=nvidia --name insightface \
    -p 20930:20920 \
    -e MXNET_CUDNN_AUTOTUNE_DEFAULT=0 \
    -e PYTHONIOENCODING=utf-8 \
    --volumes-from ibbd-data \
    -v /var/www/face_models:/models \
    -v `pwd`:/faces \
    -w /faces \
    $cmd
