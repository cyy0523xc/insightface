# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月04日 星期三 10时46分15秒
# 模拟请求
import requests
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


def image_to_base64(image_path):
    img = Image.open(image_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    return str(base64.b64encode(byte_data), encoding='utf8')


def parse_return_image(img, faces, landmarks):
    for i in range(faces.shape[0]):
        box = faces[i].astype(np.int)
        color = (0, 0, 255)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        if landmarks is None:
            continue
        landmark5 = landmarks[i].astype(np.int)
        for l in range(landmark5.shape[0]):
            color = (0, 255, 0) if l == 0 or l == 3 else (0, 0, 255)
            cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    return img


url = 'http://192.168.80.244:20920/detect_image'
body = {
    'pic': image_to_base64('lldq01.jpeg')
}
print(body['pic'][:20])
data = requests.post(url, json=body).json()

img = cv2.imread('lldq01.jpeg')
faces = np.asarray(data['data']['faces'][0])
print(faces.shape)
landmarks = np.asarray(data['data']['landmarks'])
img = parse_return_image(img, faces, landmarks)
cv2.imwrite('test.jpg', img)
