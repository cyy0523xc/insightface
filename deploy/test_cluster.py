# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年03月30日 星期六 17时53分11秒
import os
import requests

host = '192.168.80.241:20920'
url = "http://%s/cluster" % host


def send_path_dir(path_dir, k):
    body = {
        'path_dir': path_dir,
        'algo': 'dbscan',
        'k': k,
    }
    print(path_dir)
    res = requests.post(url, json=body).json()
    print(res)


def parse_dir(mp4_dir, k):
    for fn in os.listdir(mp4_dir):
        path_dir = os.path.join('/var/www/tmp/faces/tmp/', fn)
        send_path_dir(path_dir, k)


if __name__ == '__main__':
    parse_dir('/var/www/tmp/faces/videos', 2)
    parse_dir('/var/www/tmp/faces/not_match_videos', 3)
