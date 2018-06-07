#coding=utf-8
import tensorflow as tf
import os
import argparse
from tensorflow.python.framework import graph_util
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import tensorflow as tf
import sys
from tensorflow.contrib import slim
sys.path.insert(0, '/home/lol/DeepLearn/models/research/slim/')
from nets import nets_factory
from datasets import dataset_classification
from preprocessing import preprocessing_factory
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import monitored_session
from tensorflow.python.summary import summary
from tensorflow.python.ops import state_ops
import numpy as np
import math
import random
import time
import cv2
import imutils


def use_tensorflow_get_feature(base_path):
    
    checkpoint = tf.train.get_checkpoint_state('/home/lol/DeepLearn/Tensorflow-Ipynb/logs/')

    input_checkpoint = checkpoint.model_checkpoint_path

    network_fn = nets_factory.get_network_fn('resnet_v1_50', num_classes=100000, is_training=False)

    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                    shape=[None, 224,
                                        224, 3])
    network_fn(placeholder)

    saver = tf.train.Saver()

    sess = tf.Session()

    saver.restore(sess, input_checkpoint)
        
    image_preprocessing_fn = preprocessing_factory.get_preprocessing('resnet_v1_50', is_training=False)

    def format_feature(class_name, feature, image_filepath):
        return [feature, class_name, image_filepath]

    class_name_and_path_list = [[floder, os.path.join(base_path, floder)] for floder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, floder))]
    
    max_num = 0
    for class_name_and_path in class_name_and_path_list:
        image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith('.jpg')]
        max_num = max_num + len(image_path_list)
        
    now_num = 0
    last_num = 0
    last_time = time.time()
    for class_name_and_path in class_name_and_path_list:
        if os.path.exists(os.path.join(class_name_and_path[1], 'tensorflow-resnet-50_feature.npy')):
            continue
        image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith('.jpg')]
        feature_list = []
        image_list = []
        for image_path in image_path_list:
            image = imutils.opencv2matplotlib(cv2.imread(image_path))
            image_list.append(image_preprocessing_fn(image, 224, 224))
        
        image_list = sess.run(image_list)

        result_feature = sess.run("resnet_v1_50/pool5:0", feed_dict={'input:0': image_list})

        for idx, featrue in enumerate(result_feature):
            feature_list.append(format_feature(feature=featrue,
                                            class_name=class_name_and_path[0], image_filepath=image_path_list[idx]))
        now_num = now_num + len(image_path_list)
        print '正在提取中...%d/%d (%.2f/sec)' % (now_num, max_num, (now_num - last_num) / float(time.time() - last_time))
        last_num = now_num
        last_time = time.time()
        np.save(os.path.join(class_name_and_path[1], 'tensorflow-resnet-50_feature.npy'), feature_list)

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
        for t in npy:
            feature_list.append(t[0])
            class_name_list.append(t[1])
            file_path_list.append(t[2])
            now_num = now_num + 1
        print '正在提取中...%d / %d' % (now_num, max_num)

    np.save(os.path.join(base_path, 'tensorflow-feature.npy'), feature_list)
    np.save(os.path.join(base_path, 'tensorflow-class_name.npy'), class_name_list)
    np.save(os.path.join(base_path, 'tensorflow-file_path.npy'), file_path_list)

def help():
    print '用法: -f [图片路径] [选项]... [选项]...'
    print ''
    print '将文件夹内的指定格式的图片文件进行特征提取，并保存成npy格式的文件，保存格式为[feature, class_name, image_filepath]'
    print ''
    print '必选参数.'
    print '-f 指定图片所在文件夹路径'
    print ''

if __name__ == '__main__':
    
    import sys
    import getopt

    base_path = None
    prefix='resnet-50'
    layer='flatten0_output'

    opts, args = getopt.getopt(sys.argv[1:], 'f:p:l:h')

    for op, value in opts:
        if op == '-f':
            base_path = value
        elif op == '-h':
            help()
    
    if base_path is None:
        help()
        print '必须使用 -f 指定图片路径'
    else:
        use_tensorflow_get_feature(base_path)

