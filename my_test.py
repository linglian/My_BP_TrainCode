# coding=utf-8
import numpy as np
import os

base_path = '/home/lol/训练数据/加强/'

floder_list = [os.path.join(base_path, floder) for floder in os.listdir(base_path) if
               os.path.isdir(os.path.join(base_path, floder))]


feature_list = []
class_name_list = []
file_path_list = []

for i in floder_list:

    npy_class_name_list = np.load(os.path.join(i, 'tensorflow-class_name.npy'))
    npy_file_path_list = np.load(os.path.join(i, 'tensorflow-file_path.npy'))
    npy_feature_list = np.load(os.path.join(i, 'tensorflow-feature.npy'))

    print '正在读取并解析{}'.format(i) 
    for idx in range(len(npy_feature_list)):
        feature_list.append(npy_feature_list[idx])
        class_name_list.append(npy_class_name_list[idx])
        file_path_list.append(npy_file_path_list[idx])

np.save(os.path.join(base_path, 'tensorflow-feature.npy'), feature_list)
np.save(os.path.join(base_path, 'tensorflow-class_name.npy'), class_name_list)
np.save(os.path.join(base_path, 'tensorflow-file_path.npy'), file_path_list)
