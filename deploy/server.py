# -*- coding: utf-8 -*-
#
# 启动http服务
# Author: alex
# Created Time: 2019年03月26日 星期二 17时40分55秒
import cv2
import numpy as np
import face_model
from imutils.paths import list_images

model_path = ''


class Config:
    """模型配置参数"""
    model = '/models/model-r100-ii/model,0'
    # ga_model = '/models/gamodel-r50/model,0'
    ga_model = ''
    image_size = "112,112"
    gpu = 0    # gpu id
    det = 0    # mtcnn option, 1 means using R+O, 0 means detect from begining
    flip = 0   # whether do lr flip aug
    threshold = 1.24   # ver dist threshold
    num_worker = 2     #


def get_model():
    # TODO 如果放到全局变量，则会导致错误
    # mxnet.base.MXNetError: [11:56:09] src/storage/storage.cc:147:
    #   Unimplemented device 0
    # https://github.com/deepinsight/insightface/issues/415
    config = Config()
    config.model = '/models/%s/model,0' % model_path
    return face_model.FaceModel(config)


def compare(path1, path2):
    model = get_model()

    # 获取第一个图片的特征
    image1 = cv2.imread(path1)
    aligned1 = model.get_one_aligned(image1)
    if aligned1 is None:
        raise Exception('%s has no face!' % path1)
    f1 = model.get_feature(aligned1)

    image2 = cv2.imread(path2)
    aligned2 = model.get_one_aligned(image2)
    if aligned1 is None:
        raise Exception('%s has no face!' % path1)
    f2 = model.get_feature(aligned2)
    dist = np.sum(np.square(f1-f2))
    sim = np.dot(f1, f2.T)
    return {
        'dist': dist,
        'sim': sim,
    }


def compare_one2dir(path_file, path_dir):
    model = get_model()

    # 计算path_file对应的feature
    image = cv2.imread(path_file)
    feature = model.get_feature_by_image(image)
    if feature is None:
        raise Exception('%s has no face!' % path_file)

    # 获取文件夹图片的特征
    image_files = list_images(path_dir)
    image_files = sorted(list(image_files))
    data = []
    feature_t = feature.T
    for path in image_files:
        image = cv2.imread(path)
        f = model.get_feature_by_image(image)
        dist = float(np.sum(np.square(f-feature)))
        sim = float(np.dot(f, feature_t))
        data.append((dist, sim, path))

    data = sorted(data, key=lambda x: x[0])
    return data


def detect(path):
    image = cv2.imread(path)
    model = get_model()
    bboxes, points = model.detect(image)
    print('shape: ', bboxes.shape, points.shape)
    return {
        'bboxes': bboxes.tolist(),
        'points': [p.reshape((2, -1)).T.tolist() for p in points],
    }


if __name__ == '__main__':
    import sys
    from fireRest import API, app
    model_path = sys.argv[1]
    API(detect)
    API(compare)
    API(compare_one2dir)
    app.run(port=20920, host='0.0.0.0', debug=True)
