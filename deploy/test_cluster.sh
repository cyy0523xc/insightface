#!/bin/bash
# 
# 
# Author: alex
# Created Time: 2019年03月27日 星期三 14时24分40秒

curl -XPOST 192.168.80.241:20920/cluster \
    -d '{"path_dir": "'"$1"'"}'
