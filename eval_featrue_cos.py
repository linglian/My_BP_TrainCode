#coding=utf-8

import numpy as np
import sys
import os
import cv2

base_path = '/home/lol/训练数据/深度学习-布料库-加强版-整理/'

base_featrue_list = np.load(os.path.join(base_path, 'tensorflow-feature.npy'))
base_class_name_list = np.load(os.path.join(base_path, 'tensorflow-class_name.npy'))
base_file_path_list = np.load(os.path.join(base_path, 'tensorflow-file_path.npy'))

test_path = '/home/lol/训练数据/布料数据特征/'

test_featrue_list = np.load(os.path.join(test_path, 'tensorflow-feature.npy'))
test_class_name_list = np.load(os.path.join(test_path, 'tensorflow-class_name.npy'))
test_file_path_list = np.load(os.path.join(test_path, 'tensorflow-file_path.npy'))

def getDistOfL2(form, to):
    return cv2.norm(form, to, normType=cv2.NORM_L2)

def getDistOfSquare(form, to):
    return np.sqrt(np.sum(np.square(form - to)))

def getDistOfHash(f, t):
    return f[0].__sub__(t[0])

def getDistOfCos(f, t):
    up = np.sum(np.multiply(f, t))
    ff = np.sqrt(np.sum(np.multiply(f, f)))
    tt = np.sqrt(np.sum(np.multiply(t, t)))
    down = ff * tt
    return up / down

def getRank(test_featrue):
    rank_list = []

    for i in base_featrue_list:
        rank_list.append(getDistOfCos(test_featrue, i))

    rank_list.sort()

    print rank_list
    return rank_list


