# coding=utf-8
import multiprocessing
from multiprocessing.connection import Listener
from multiprocessing import Pool
import sys
import getopt
import traceback
sys.path.insert(0, '/home/lol/anaconda2/lib/python2.7/site-packages')
import falconn
import os
import cv2
import numpy as np
import time
from PIL import Image
import gc
from sklearn.decomposition import PCA

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='test_train_Server.log',
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 公共区
path = None
is_pool = True

# 提取特征区
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
sys.path.insert(0, '/home/lee/DeepLearn/models/research/slim/')
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

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

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
        feature_list = my_feature[find_list]
        return class_name_list, file_path_list, feature_list, featrue
    else:
        return None, None


# 客户端发来的请求进行处理(最好需要几个就设置多少k，不然影响速度)
# return 图片特征，最近的数组(k * max_img长度的数组, 相似度从近到远)
def make_work(conn):
    global my_arr, my_id, big_class
    try:
        msg=conn.recv()
        logging.info(msg)
        opts, args=getopt.getopt(msg, 'f:zt:k:sp', ['help', 'train'])
        img=None
        rank=20
        img_type=0
        is_save=True
        msg=[]
        filepath=None
        is_pcl=False
        for op, value in opts:
            if op == '-f':
                img=value
            elif op == '-z':
                return 'Close'
            elif op == '-k':
                rank=int(value)
            elif op == '-s':
                is_save=False
            elif op == '--train':
                return "train"
            elif op == '-p':
                is_pcl=True
            elif op == '-t':
                img_type=value
            elif op == '--help':
                msg.append(' ')
                msg.append('Usage:')
                msg.append('  Client [options]')
                msg.append(' ')
                msg.append('General Options:')
                msg.append('-f <path>\t\t Set test image path')
                msg.append('-z \t\t\t Close server')
                msg.append('-k <number>\t\t Set rank')
                msg.append('-s \t\t\t No Save image of rank K')
                msg.append(
                    '-t <number>\t\t Set image type if you want to know test type')
                return msg
        if img is None:
            msg.append('Must set Image Path use -f')
            return msg
        ti_time= time.time()
        class_name_list, file_path_list, feature_list, featrue = find_k_FeatureHash(path, img, rank)
        msg.append('Next')
        msg.append(class_name_list)
        msg.append(file_path_list)
        msg.append(feature_list)
        msg.append(featrue)
        msg.append('Test Image Spend Time: %.2lf s' %
                    (time.time() - ti_time))
        return msg
    except EOFError:
        logging.info('Connection closed')
        return None


# 运行Server，一直监听接口
def run_server(address, authkey):
    serv = Listener(address, authkey =authkey)
    while True:
        try:
            client= serv.accept()
            msg= make_work(client)
            if msg == 'Close': # 关闭监听
                serv.close()
                return "Close"
            else:
                client.send(msg)
        except Exception:
            traceback.print_exc()
    serv.close()


if __name__ == '__main__':
    opts, args= getopt.getopt(sys.argv[1:], 'f:x:p:k:b:d:t:')
    server_id= 99
    for op, value in opts:
        if op == '-f': # 设置四个NPY文件所在文件夹路径
            path = value
        elif op == '-p': # 设置运行时的id，同于通信
            server_id = int(value)
        elif op == '-b': # 设置最大PCA倍数
            beishu = int(value)
        elif op == '-d': # 设置最大图片数量
            dim = int(value)
        elif op == '-x': # 设置mxnet/python所在路径
            mxnetpath = value
            sys.path.insert(0, mxnetpath)
        elif op == '-t': # 设置增强数量
            tilesPerImage = int(value)
    if path is None:
        print '必须使用 -f 输入model_path用来指定三个npy文件路径'
    else:
        while True:
            logging.info('Start Run')
            run_server('/usr/local/server%d.temp' % server_id, b'lee123456')
            logging.info('Stop Run')
