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


tf.logging.set_verbosity(tf.logging.INFO)

checkpoint = tf.train.get_checkpoint_state('/home/lol/DeepLearn/Tensorflow-Ipynb/logs/')

input_checkpoint = checkpoint.model_checkpoint_path

print input_checkpoint

import cv2
import imutils

network_fn = nets_factory.get_network_fn('resnet_v1_50', num_classes=100000, is_training=False)

placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                shape=[1, 224,
                                    224, 3])
network_fn(placeholder)

sess = tf.Session()

saver = tf.train.Saver()

saver.restore(sess, input_checkpoint)

image_preprocessing_fn = preprocessing_factory.get_preprocessing('resnet_v1_50', is_training=False)
   
def find_k_softmax(image_file_path, k):
    
    def getSoftmax(image_filepath):
        import cv2
        import numpy as np

        image = imutils.opencv2matplotlib(cv2.imread(image_filepath))

        image = image_preprocessing_fn(image, 224, 224)

        image = image.eval(session=sess)

        result_feature = sess.run('resnet_v1_50/predictions/Softmax:0', feed_dict={'input:0': [image]})

        return np.ravel(result_feature)
    
    softmax = getSoftmax(image_file_path)
    #find_k_list = np.argsort(softmax)[-k:]
    return softmax

def eval_softmax(base_path, k = [1, 5, 10], image_format='.webp'):
    def test_softmax(bashpath, k, image_format):

        class_name_and_path_list = [[floder, os.path.join(base_path, floder)] for floder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, floder))]

        all_bad_list = []
        if isinstance(k, int):
            k = [k]
            
        for tk in k:
            tk = int(tk)
            num = 0
            now_num = 0
            bad_list = []
            random.shuffle(class_name_and_path_list)
            
            my_sum = 0

            for class_name_and_path in class_name_and_path_list[:50]:
                image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith(image_format)]
                my_sum = my_sum + len(image_path_list)
            
            print '一共要检测{}个图片'.format(my_sum)

            for class_name_and_path in class_name_and_path_list[:50]:

                image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith(image_format)]
                class_name = class_name_and_path[0]
                for image_path in image_path_list:
                    find_k_list = find_k_softmax(image_file_path=image_path, k=tk)
                    now_num = now_num + 1
                    find_k_list = find_k_list.argsort()[-tk:]
                    #print find_k_list, int(class_name)
                    if len(np.argwhere(find_k_list == int(class_name))) is not 0:
                        num = num + 1

                print 'Rank=%d时的正确率为%.02f(%d/%d)' % (tk, float(num) / now_num, num, my_sum)
            all_bad_list.append(bad_list)
        return all_bad_list
    all_bad_list = test_softmax(base_path, k, image_format)
    np.save('./all_bad_list.npy', all_bad_list)

def help():
    print '用法: -f [图片路径] [选项]... [选项]...'
    print ''
    print '将文件夹内的指定格式的图片文件进行FeatureHash测试'
    print ''
    print '必选参数.'
    print '-f 指定图片所在文件夹路径，必须使用tidy_image进行格式处理'
    print ''
    print '可选参数.'
    print '-r 指定Rank 默认为[1, 5, 10]'
    print '-t 指定图片格式'
    print ''

if __name__ == '__main__':
    
    import sys
    import getopt

    base_path = None
    rank = [1, 5, 10]
    image_format = '.webp'

    opts, args = getopt.getopt(sys.argv[1:], 'f:r:t:h')

    for op, value in opts:
        if op == '-f':
            base_path = value
        elif op == '-r':
            rank = value
        elif op == '-t':
            image_format = value
        elif op == '-h':
            help()
    
    if base_path is None:
        help()
        print '必须使用 -f 指定图片路径'
    else:
        eval_softmax(base_path, rank, image_format)

