# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月26日 星期四 10时44分21秒
import os
import cv2
import json
from demo import get_detector, face_detect


def parse_video(video_path, output_path, rate=1):
    vc = cv2.VideoCapture(video_path)
    if vc.isOpened() is False:
        raise Exception('video open false!')

    index = 0
    mod = int(vc.get(cv2.CAP_PROP_FPS))*rate
    detector = get_detector()
    while True:
        rval, frame = vc.read()
        if rval is False:
            break
        if index % mod != 0:
            continue
        index += 1
        faces, landmarks = face_detect(detector, frame)
        if len(faces) < 1:
            continue
        fn = video_path.split('/')[-1]
        fn = fn.split('.')[0]
        for i, (face, lm) in enumerate(zip(faces, landmarks)):
            # 保存头像文件
            img_fn = "%s-%03d-%d.jpg" % (fn, index, i)
            path = os.path.join(output_path, fn, 'images', img_fn)
            x, y, xb, yb, _ = face
            img = frame[int(y):int(yb), int(x):int(xb)]
            cv2.imwrite(path, img)
            # 保存对应数据文件
            json_fn = "%s-%03d-%d.json" % (fn, index, i)
            path = os.path.join(output_path, fn, 'images', json_fn)
            with open(path, 'w') as w:
                json.dump({
                    'face': face,
                    'landmark': lm,
                }, w)


if __name__ == '__main__':
    import sys
    parse_video(sys.argv[1], sys.argv[2])
