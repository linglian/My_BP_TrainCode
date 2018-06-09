
#coding=utf-8
import numpy as np
import os

base_path = '/media/lol/训练数据/深度学习-布料库-加强版-整理/'

class_name_and_path_list = [[floder, os.path.join(base_path, floder)] for floder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, floder))]
    
max_num = 0
for class_name_and_path in class_name_and_path_list:
    image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith('.jpg')]
    max_num = max_num + len(image_path_list)
    
now_num = 0

feature_list = []
class_name_list = []
file_path_list = []

for class_name_and_path in class_name_and_path_list:
    image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith('.jpg')]
    npy = np.load(os.path.join(class_name_and_path[1], 'tensorflow-resnet-50_feature.npy'))
    for t in range(0, len(npy[0][0]) / 2048):
        feature_list.append(npy[0][0][2048 * t: 2048 * (t + 1)])
        #print np.array(npy[0][0][2048 * t: 2048 * (t + 1)]).shape
        print npy[0][2]
        print int(npy[0][1])
        class_name_list.append(npy[0][1])
        file_path_list.append(npy[0][2])
        now_num = now_num + 1
    print '正在提取中...%d / %d' % (now_num, max_num)

np.save(os.path.join(base_path, 'tensorflow-feature.npy'), feature_list)
np.save(os.path.join(base_path, 'tensorflow-class_name.npy'), class_name_list)
np.save(os.path.join(base_path, 'tensorflow-file_path.npy'), file_path_list)