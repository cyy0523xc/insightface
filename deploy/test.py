import face_model
import argparse
import cv2
import sys
import numpy as np

"""
python test.py --model ../models/gamodel-r50/model,0 \
    --ga-model ../models/gamodel-r50/model,0 \
    --image-file ../tests/lldq01.jpeg
"""
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-file', default='', help='path to image file.')
parser.add_argument('--image-cmp-file', default='', help='path to image file.')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread(args.image_file)
bboxes, points = model.detect(img)
print('image shape: ', img.shape, " person count: ", len(bboxes))

features = []
for index, bbox, point in zip(range(len(bboxes)), bboxes, points):
    print('Person: %d' % index, '*'*40)
    print('bbox: ', bbox)
    point = point.reshape((2, 5)).T
    print('points: ', point)
    aligned = model.get_aligned(img, bbox, point)
    print('aligned image shape: ', aligned.shape)
    f1 = model.get_feature(aligned)
    features.append(f1)
    print("feature: ", f1[0:10])
    gender, age = model.get_ga(aligned)
    print('gender: ', gender)
    print('age: ', age)

print('*'*60)
img = cv2.imread(args.image_cmp_file)
img = model.get_one_aligned(img)
f2 = model.get_feature(img)
dists = [np.sum(np.square(f1-f2)) for f1 in features]
print(min(dists), dists)
simes = [np.dot(f1, f2.T) for f1 in features]
print(simes)
