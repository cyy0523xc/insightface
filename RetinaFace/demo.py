# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月03日 星期二 14时11分56秒
import cv2
import numpy as np
from retinaface import RetinaFace

thresh = 0.8
count = 1
gpuid = 0
detector = RetinaFace('/models/R50', 0, gpuid, 'net3')


def face_detect(image_path, out_path):
    scales = [1024, 1980]
    img = cv2.imread(image_path)
    print(img.shape)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.0
    # if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    print('im_scale', im_scale)

    scales = [im_scale]
    flip = False

    for c in range(count):
        faces, landmarks = detector.detect(img, thresh,
                                           scales=scales, do_flip=flip)
        print(c, faces.shape, landmarks.shape)

    if faces is None:
        return

    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
        # print('score', faces[i][4])
        box = faces[i].astype(np.int)
        # color = (255,0,0)
        color = (0, 0, 255)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        if landmarks is None:
            continue
        landmark5 = landmarks[i].astype(np.int)
        # print(landmark.shape)
        for l in range(landmark5.shape[0]):
            color = (0, 255, 0) if l == 0 or l == 3 else (0, 0, 255)
            cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    cv2.imwrite(out_path, img)


if __name__ == '__main__':
    from fireRest import API, app
    API(face_detect)
    app.run(port=20920, host='0.0.0.0', debug=True)
