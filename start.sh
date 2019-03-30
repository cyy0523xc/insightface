#!/bin/bash
# 
# 启动
# Author: alex
# Created Time: 2019年03月26日 星期二 15时13分07秒
"""
python3 test.py --model /models/gamodel-r50/model,0 \
    --ga-model /models/gamodel-r50/model,0 \
    --image-file ../tests/lldq01.jpeg
"""
docker run --rm -ti --runtime=nvidia --name insightface \
    -p 20920:20920 \
    -v /var/www/face_models:/models \
    -v /var/www/tmp/faces:/var/www/tmp/faces \
    -v /var/www/github.com/insightface:/faces \
    -w /faces \
    "$1" /bin/bash
