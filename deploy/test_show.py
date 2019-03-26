# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年03月26日 星期二 17时54分50秒
import cv2
import requests

host = '192.168.80.241:20920'


def detect(path):
    """
    Args:
        path: 图像路径
    """
    url = "http://%s/detect" % host
    body = {
        'path': path
    }
    res = requests.post(url, json=body).json()
    print(res)

    res = res['data']
    image = cv2.imread(path)
    for x, y, xb, yb, score in res['bboxes']:
        x, y, xb, yb = int(x), int(y), int(xb), int(yb)
        cv2.rectangle(image, (x, y), (xb, yb), (0, 0, 255), thickness=1)
        cv2.putText(image, "%.3f" % score, (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 255, 0), 1)

    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    import fire
    fire.Fire(detect)
