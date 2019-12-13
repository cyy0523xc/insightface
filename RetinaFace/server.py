# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年12月11日 星期三 18时11分42秒
import time
import numpy as np
from retinaface import RetinaFace
from image import base64_cv2

model_path = '/models/R50'
g_model = None


def init_model(gpuid=0):
    global g_model
    g_model = RetinaFace(model_path, 0, gpuid, 'net3')


def detect_images(b64_list, thresh=0.8):
    """人脸检测
    :return [{"bboxes": [], "landmarks": []}]
    """
    start = time.time()
    data = []
    for img in b64_list:
        img = base64_cv2(img)
        bboxes, landmarks = face_detect(img, thresh=thresh)
        data.append({
            "bboxes": bboxes.tolist(),
            "landmarks": landmarks.tolist(),
        })

    print('===> Time: ', time.time() - start, '  Total: ', len(b64_list))
    return data


def face_detect(img, thresh=0.8):
    scales = [1024, 1980]
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

    scales = [im_scale]
    flip = False
    faces, landmarks = g_model.detect(img, thresh,
                                      scales=scales, do_flip=flip)
    if faces is None:
        return np.array([]), np.array([])

    # print(faces.shape, landmarks.shape)
    # print('find', faces.shape[0], 'faces')
    return faces, landmarks


if __name__ == '__main__':
    from fireRest import API, app
    init_model()
    API(detect_images)
    app.run(port=20920, host='0.0.0.0', debug=True, threaded=False)
