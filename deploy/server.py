# -*- coding: utf-8 -*-
#
# 启动http服务
# Author: alex
# Created Time: 2019年03月26日 星期二 17时40分55秒
import cv2
import face_model

model_path = ''


class Config:
    """模型配置参数"""
    model = '/models/model-r100-ii/model,0'
    ga_model = '/models/gamodel-r50/model,0'
    image_size = "112,112"
    gpu = 0    # gpu id
    det = 0    # mtcnn option, 1 means using R+O, 0 means detect from begining
    flip = 0   # whether do lr flip aug
    threshold = 1.24   # ver dist threshold
    num_worker = 2     #


def get_model():
    # TODO 如果放到全局变量，则会导致错误
    # mxnet.base.MXNetError: [11:56:09] src/storage/storage.cc:147: Unimplemented device 0
    # https://github.com/deepinsight/insightface/issues/415
    config = Config()
    config.model = '/models/%s/model,0' % model_path
    return face_model.FaceModel(config)


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
    app.run(port=20920, host='0.0.0.0')
