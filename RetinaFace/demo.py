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

default_model_path = '/models/R50'


def detect_file(image_path, out_path='out.jpg',
                model_path=default_model_path, thresh=0.8, gpuid=0):
    """人脸检测（输入输出都是图片）
    :param image_path 输入图片相对路径
    :param out_path 输出图片相对地址
    :return
    """
    img = cv2.imread(image_path)
    print(img.shape)
    faces, landmarks = face_detect(img, model_path=model_path)
    out_img = parse_return_image(img, faces, landmarks)
    cv2.imwrite(out_path, out_img)
    return


def detect_image(pic, model_path=default_model_path, return_image=False):
    """人脸检测（输入的是base64编码的图像）
    :param pic 图片对象使用base64编码
    :param return_image 是否返回图片对象，base64编码，默认值为false
    :return 当return_image=false时，返回值为{'faces': [], 'landmarks': []}，其中:
        faces是人脸边框，landmarks是人脸的5个关键点
        当return_image=true时，返回值为{'pic': 图片对象}，pic值也是base64编码
    """
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
    faces, landmarks = face_detect(img, model_path=model_path)
    if return_image is False:
        # 返回数据
        return {
            'faces': faces,
            'landmarks': landmarks
        }

    # 返回图像
    out_img = parse_return_image(img, faces, landmarks)
    out_img = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    output_buffer = io.BytesIO()
    out_img.save(output_buffer, format='JPEG')
    binary_data = output_buffer.getvalue()
    return {
        'pic': str(base64.b64encode(binary_data), encoding='utf8')
    }


def face_detect(img, model_path=default_model_path, thresh=0.8, gpuid=0):
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
    return faces, landmarks


def parse_return_image(img, faces, landmarks):
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
    app.run(port=20920, host='0.0.0.0', debug=True)
