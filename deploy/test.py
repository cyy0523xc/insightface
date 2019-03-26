import face_model
import argparse
import cv2
import sys
import numpy as np

"""
python test.py --model ../models/gamodel-r50/model,0 \
    --ga-model ../models/gamodel-r50/model,0
"""
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-file', default='', help='path to image file.')
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
print('image shape: ', img.shape)
img, bbox, points = model.get_input(img)
print('image shape: ', img.shape)
print('bbox: ', bbox)
print('points: ', points)
f1 = model.get_feature(img)
print("feature: ", f1[0:10])
gender, age = model.get_ga(img)
print('gender: ', gender)
print('age: ', age)
sys.exit(0)

img = cv2.imread('/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54733.png')
f2 = model.get_feature(img)
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
