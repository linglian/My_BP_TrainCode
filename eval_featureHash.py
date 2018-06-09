#coding=utf-8
import tensorflow as tf
import os
import argparse
from tensorflow.python.framework import graph_util
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import tensorflow as tf
import sys
import time
from tensorflow.contrib import slim
sys.path.insert(0, '/home/lol/DeepLearn/models/research/slim/')
from nets import nets_factory
from datasets import dataset_classification
from preprocessing import preprocessing_factory
from tensorflow.contrib.slim import evaluation
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import monitored_session
from tensorflow.python.summary import summary
import numpy as np
import falconn
import random
import imutils
import cv2

checkpoint = tf.train.get_checkpoint_state('/home/lol/DeepLearn/Tensorflow-Ipynb/logs/')

input_checkpoint = checkpoint.model_checkpoint_path

global is_need_init_find_k_FeatureHash
global falconn_query, mxnet_feature_extractor
global my_feature, my_class_name, my_file_path

is_need_init_find_k_FeatureHash = True

network_fn = nets_factory.get_network_fn('resnet_v1_50', num_classes=100000, is_training=False)

placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                shape=[1, 224,
                                    224, 3])
network_fn(placeholder)

saver = tf.train.Saver()

sess = tf.Session()

saver.restore(sess, input_checkpoint)
    
image_preprocessing_fn = preprocessing_factory.get_preprocessing('resnet_v1_50', is_training=False)

img_pla = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], name='img')

image_preprocessing = image_preprocessing_fn(img_pla, 224, 224)

def find_k_FeatureHash(model_path, image_file_path, k):
    global is_need_init_find_k_FeatureHash
    global falconn_query, mxnet_feature_extractor
    global my_feature, my_class_name, my_file_path
    
    def getFeature(image_filepath):

        img = cv2.imread(image_filepath)
        if img is not None:
            image = imutils.opencv2matplotlib(img)

            image = sess.run(image_preprocessing, feed_dict={'img:0': image})

            result_feature = sess.run("resnet_v1_50/pool5:0", feed_dict={'input:0': [image]})

            return np.ravel(result_feature)
        else:
            return None
    
    if is_need_init_find_k_FeatureHash:
        print '正在进行初始化...'
        def init_falconn():
            dim = 2048
            # 获得数组
            my_feature = np.load(os.path.join(model_path, 'tensorflow-feature.npy'))
            print my_feature.shape
            my_class_name = np.load(os.path.join(model_path, 'tensorflow-class_name.npy'))
            print my_class_name.shape
            my_file_path = np.load(os.path.join(model_path, 'tensorflow-file_path.npy'))
            print my_file_path.shape
            # 获取数组数量
            trainNum=len(my_feature)
            # 获得默认参数
            p=falconn.get_default_parameters(trainNum, dim)
            t=falconn.LSHIndex(p)
            dataset = my_feature
            # 生成hash
            t.setup(dataset)
            q = t.construct_query_pool()
            return my_feature, my_class_name, my_file_path, q
        my_feature, my_class_name, my_file_path, falconn_query = init_falconn()
        is_need_init_find_k_FeatureHash = False
        print '初始化结束...'
    featrue = getFeature(image_file_path)
    if featrue is not None:
        find_list = falconn_query.find_k_nearest_neighbors(query=featrue, k=k)
        class_name_list = my_class_name[find_list]
        file_path_list = my_file_path[find_list]
        return class_name_list, file_path_list
    else:
        return None, None

def eval_featureHash(base_path, model_path, k = [1, 5, 10], image_format='.webp'):
    if isinstance(base_path, str):
        base_path = [base_path]
    def test_featureHash(base_path, model_path, k, image_format):

        class_name_and_path_list = [[floder, os.path.join(base_path, floder)] for floder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, floder))]

        all_bad_list = []
        if isinstance(k, int):
            k = [k]
            
        for tk in k:
            num = 0
            now_num = 0
            bad_list = []
            random.shuffle(class_name_and_path_list)
            tic = 0
            my_sum = 0

            for class_name_and_path in class_name_and_path_list[:100]:
                image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith(image_format)]
                my_sum = my_sum + len(image_path_list)
            
            print '一共要检测{}个图片'.format(my_sum)

            for class_name_and_path in class_name_and_path_list[:100]:

                image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith(image_format)]
                class_name = class_name_and_path[0]
                for image_path in image_path_list:
                    c, f = find_k_FeatureHash(model_path, image_file_path=image_path, k=tk)
                    if c is not None:
                        is_find = False
                        now_num = now_num + 1
                        for tc in c:
                            #print tc, class_name
                            if str(tc) == str(class_name):
                                num = num + 1
                                is_find = True
                                break
                        if is_find is False:
                            bad_list.append(image_path)
                tic += 1
            if tic % 1 == 0:
                print 'Rank=%d时的正确率为%.02f(%d/%d)' % (tk, float(num) / now_num, num, my_sum)
            all_bad_list.append(bad_list)
        return all_bad_list
    for idx, b_p in enumerate(base_path):
        print '开始进行   {}   检测'.format(b_p)
        all_bad_list = test_featureHash(b_p, model_path, k, image_format)
        np.save('./all_bad_list_{}.npy'.format(idx), all_bad_list)


def help():
    print '用法: -f [图片路径] [选项]... [选项]...'
    print ''
    print '将文件夹内的指定格式的图片文件进行FeatureHash测试'
    print ''
    print '必选参数.'
    print '-f 指定图片所在文件夹路径，必须使用tidy_image进行格式处理'
    print '-m 指定三个npy文件所在路径'
    print ''
    print '可选参数.'
    print '-r 指定Rank 默认为[1, 5, 10]'
    print '-t 指定图片格式, 默认为.webp'
    print ''

if __name__ == '__main__':
    
    import sys
    import getopt

    model_path = None
    base_path = None
    rank = [1, 5, 10]
    image_format = '.webp'

    opts, args = getopt.getopt(sys.argv[1:], 'f:m:r:t:h')

    for op, value in opts:
        if op == '-f':
            base_path = value.split(',')
        elif op == '-m':
            model_path = value
        elif op == '-r':
            rank = value
        elif op == '-t':
            image_format = value
        elif op == '-h':
            help()
    
    if model_path is None:
        help()
        print '必须使用 -m 指定三个npy所在路径'
    elif base_path is None:
        help()
        print '必须使用 -f 指定图片路径'
    else:
        eval_featureHash(base_path, model_path, rank, image_format)

