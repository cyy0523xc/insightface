# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月03日 星期二 14时11分56秒
import io
import cv2
import base64
import numpy as np
from PIL import Image
from retinaface import RetinaFace


def detect_file(image_path, out_path='out.jpg', model_path='/models/R50',
                thresh=0.8, gpuid=0):
    img = cv2.imread(image_path)
    print(img.shape)
    out_img = face_detect(img, model_path=model_path)
    cv2.imwrite(out_path, out_img)
    return


def detect_image(pic, model_path='/models/R50'):
    tmp = pic.split(',')[0]
    pic = pic[len(tmp)+1:]
    pic = base64.b64decode(pic)
    pic = Image.open(io.BytesIO(pic))
    if 'png' in tmp:   # 先转化为jpg
        bg = Image.new("RGB", pic.size, (255, 255, 255))
        bg.paste(pic, pic)
        pic = bg

    img = cv2.cvtColor(np.asarray(pic), cv2.COLOR_RGB2BGR)
    print(img.shape)
    out_img = face_detect(img, model_path=model_path)
    out_img = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    output_buffer = io.BytesIO()
    out_img.save(output_buffer, format='WEBP')
    binary_data = output_buffer.getvalue()
    return {'pic': str(base64.b64encode(binary_data), encoding='utf8')}


def face_detect(img, model_path='/models/R50', thresh=0.8, gpuid=0):
    detector = RetinaFace(model_path, 0, gpuid, 'net3')
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

    print('im_scale', im_scale)

    scales = [im_scale]
    flip = False
    faces, landmarks = detector.detect(img, thresh,
                                       scales=scales, do_flip=flip)
    print(faces.shape, landmarks.shape)

    if faces is None:
        raise Exception('no faces!')

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

    return img


if __name__ == '__main__':
    from fireRest import API, app
    # curl -XPOST localhost:20920/detect_file
    #     -d '{"image_path": "../tests/celian01.jpeg", "out_path": "out.jpg"}'
    API(detect_file)
    API(detect_image)
    app.run(port=20920, host='0.0.0.0')
