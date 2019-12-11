# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2019年09月03日 星期二 14时11分56秒
import re
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from retinaface import RetinaFace

default_model_path = '/models/R50'


def get_detector(model_path=default_model_path, gpuid=0):
    return RetinaFace(model_path, 0, gpuid, 'net3')


def detect_file(image_path, out_path='out.jpg',
                model_path=default_model_path, thresh=0.8, gpuid=0):
    """人脸检测（输入输出都是图片路径）
    :param image_path 输入图片相对路径
    :param out_path 输出图片相对地址
    :return
    """
    img = cv2.imread(image_path)
    detector = get_detector(model_path=model_path)
    faces, landmarks = face_detect(detector, img)
    out_img = parse_return_image(img, faces, landmarks)
    cv2.imwrite(out_path, out_img)
    return


def detect_image(image='', image_path='', image_type='jpg',
                 model_path=default_model_path,
                 return_data=True, return_image=False):
    """人脸检测（输入的是base64编码的图像）
    :param image 图片对象使用base64编码
    :param image_path 图片路径
    :param image_type 输入图像类型, 取值jpg或者png
    :param return_data 是否返回数据，默认为True。
        若该值为True，则返回值里会包含faces与landmarks
        faces是人脸边框，landmarks是人脸的5个关键点
    :param return_image 是否返回图片对象，base64编码，默认值为false
        当return_image=true时，返回值为{'image': 图片对象}，image值也是base64编码
    :return {'faces': [], 'landmarks': [], 'image': str}
    """
    if not image and not image_path:
        raise Exception('image参数和image_path参数必须有一个不为空')

    if image:
        # 自动判断类型
        type_str = re.findall('^data:image/.+;base64,', image)
        if len(type_str) > 0:
            if 'png' in type_str[0]:
                image_type = 'png'

        image = re.sub('^data:image/.+;base64,', '', image)
        image = base64.b64decode(image)
        image = Image.open(BytesIO(image))
        if image_type == 'png':   # 先转化为jpg
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, image)
            image = bg

        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_path)

    detector = get_detector(model_path=model_path)
    faces, landmarks = face_detect(detector, img)
    data = {}
    if return_data:
        # 返回数据
        data['faces'] = faces.tolist(),
        data['landmarks'] = landmarks.tolist()

    out_img = None
    if return_image:
        # 返回图像
        out_img = parse_return_image(img, faces, landmarks)
        out_img = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
        out_img = parse_output_image(out_img)

    return {
        'image': out_img,
        'data': data
    }


def face_detect(detector, img, thresh=0.8):
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


def parse_output_image(out_img):
    """base64字符串"""
    output_buffer = BytesIO()
    out_img.save(output_buffer, format='JPEG')
    binary_data = output_buffer.getvalue()
    return str(base64.b64encode(binary_data), encoding='utf8')


def get_demo_image(path):
    """获取演示图片"""
    img = Image.open(path)
    return {
        'image': parse_output_image(img)
    }


if __name__ == '__main__':
    from fireRest import API, app
    # curl -XPOST localhost:20920/detect_file
    #     -d '{"image_path": "../tests/celian01.jpeg", "out_path": "out.jpg"}'
    API(detect_file)
    API(detect_image)
    API(get_demo_image)
    app.run(port=20920, host='0.0.0.0', debug=True, threaded=False)
