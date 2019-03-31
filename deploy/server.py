# -*- coding: utf-8 -*-
#
# 启动http服务
# Author: alex
# Created Time: 2019年03月26日 星期二 17时40分55秒
import os
import cv2
import numpy as np
import face_model
from imutils.paths import list_images
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabaz_score


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
    data = []
    feature_t = feature.T
    for path in sorted(list(image_files)):
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


def detect_dir(path_dir):
    print(path_dir)
    model = get_model()
    data = []
    image_files = list(list_images(path_dir))
    print("images files: %d" % len(image_files))
    for path in sorted(image_files):
        image = cv2.imread(path)
        bboxes, points = model.detect(image)
        if bboxes is not None:
            bboxes = bboxes.tolist()
            points = [p.reshape((2, -1)).T.tolist() for p in points]
        data.append({
            'path': path,
            'bboxes': bboxes,
            'points': points,
        })

    return data


def cluster(path_dir, algo='kmeans', k=2, face_score=0.9995, eps=0.9):
    model = get_model()
    image_files = list_images(path_dir)
    X, aligned_images = [], []
    print('begin to detect images:')
    for path in sorted(list(image_files)):
        image = cv2.imread(path)
        bboxes, pointses = model.detect(image)
        if bboxes is None:
            continue

        aligneds = [model.get_aligned(image, bbox, points.reshape((2, -1)).T)
                    for bbox, points in zip(bboxes, pointses)
                    if bbox[4] > face_score]
        features = [model.get_feature(a) for a in aligneds]
        aligned_images += aligneds
        X += features

    # cluster
    print('aligned image shape: ', aligned_images[0].shape)
    print('begin to cluster:')
    save_dir = 'cluster_out_k%d_score_%f_eps_%f/' % (k, face_score, 0.9)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if path_dir.endswith('/'):
        path_dir = path_dir[:-1]

    save_dir += path_dir.split('/')[-1]+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if algo == 'kmeans':
        y_pred = KMeans(n_clusters=k).fit_predict(X)
    elif algo == 'dbscan':
        y_pred = DBSCAN(metric="euclidean", eps=eps).fit_predict(X)

    for i, img, y in zip(range(len(y_pred)), aligned_images, y_pred):
        path = save_dir + ("class_%d/" % y)
        if not os.path.isdir(path):
            os.mkdir(path)

        img = np.transpose(img, (1, 2, 0))
        cv2.imwrite(path+'%d.jpg' % i, img)

    X = np.array(X)
    uniq_y = set(y_pred)
    count = max(uniq_y) + 1
    count_x = [len(X[y_pred==i]) for i in range(count)]
    score, dist = 0, []
    if count > 1:
        score = calinski_harabaz_score(X[y_pred>=0], y_pred[y_pred>=0])
        dist = cal_set_dist(count, X[y_pred>=0], y_pred[y_pred>=0])
    return {
        'score': score,
        'count': int(count),
        'count_x': count_x,
        'dist': dist,
    }


def cal_set_dist(count, X, y):
    """计算不同类别之间的距离"""
    groups = [X[y==i] for i in range(count)]
    data = {}
    for i in range(count):
        for j in range(i+1, count):
            key = "%d-%d" % (i, j)
            data[key] = dist_2set(groups[i], groups[j])

    return data


def dist_2set(group1, group2):
    dist = []
    for i in group1:
        i_g2 = [np.sum(np.square(j - i)) for j in group2]
        dist.append(min(i_g2))

    return float(min(dist)), float(np.average(dist)), float(max(dist))


if __name__ == '__main__':
    from fireRest import API, app
    API(detect)
    API(detect_dir)
    API(compare)
    API(cluster)
    API(compare_one2dir)
    app.run(port=20920, host='0.0.0.0', debug=True)
