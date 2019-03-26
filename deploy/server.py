# -*- coding: utf-8 -*-
#
# 启动http服务
# Author: alex
# Created Time: 2019年03月26日 星期二 17时40分55秒
import cv2
import face_model


class Config:
    """模型配置参数"""
    model = '/models/gamodel-r50/model,0'
    ga_model = '/models/gamodel-r50/model,0'
    image_size = "112,112"
    gpu = 0    # gpu id
    det = 0    # mtcnn option, 1 means using R+O, 0 means detect from begining
    flip = 0   # whether do lr flip aug
    threshold = 1.24   # ver dist threshold


model = face_model.FaceModel(Config())


def detect(path):
    image = cv2.imread(path)
    bboxes, points = model.detect(image)
    return {
        'bboxes': bboxes.tolist(),
        'points': points.reshape((2, -1)).T.tolist(),
    }


if __name__ == '__main__':
    from fireRest import API, app
    API(detect)
    app.run(port=20920, host='0.0.0.0')
