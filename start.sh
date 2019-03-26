#!/bin/bash
# 
# 启动
# Author: alex
# Created Time: 2019年03月26日 星期二 15时13分07秒
docker run --rm -ti --runtime=nvidia --name insightface \
    -v /var/www/insightface/models:/models \
    -v /var/www/github.com/insightface:/faces \
    "$1" /bin/bash
