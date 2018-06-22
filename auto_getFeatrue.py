
#coding=utf-8
import os
import sys
import getopt

reload(sys)
sys.setdefaultencoding('utf-8')

# 数据增强区
from tools import data_augmentation
from tools import run_tensorflow_get_feature

num = 0

# 数据整理
def run_tidy(file_path, save_floder_path, file_format='.jpg'):
    global num
    from tools import file_tools
    def let_do_it(file_path):
        file_name = file_tools.getFileName(file_path)
        class_name = file_tools.getFloderOfFile(file_path, 1)
        file_list = []
        if os.path.exists(os.path.join(save_floder_path, class_name)):
            file_list = os.listdir(os.path.join(save_floder_path, class_name))
        if os.path.exists(os.path.join(save_floder_path, class_name, file_name + file_format)) is not True:
            num = num +  1
            file_tools.copy_file(file_path, os.path.join(save_floder_path, class_name, file_name + file_format))
    file_tools.traverse_floder(file_path, let_do_it, check_file_format=file_format)

# 提取特征
def run_tensorflow(base_path, save_path=None):
    if save_path is None:
        save_path = base_path
    run_tensorflow_get_feature.use_tensorflow_get_feature(base_path,save_path)

first_path = u'/media/lee/data/macropic/newp/new/'
data_augmentation_save_path = u'/media/lee/data/macropic/newp/new_数据加强版/'
tidy_save_path = u'/media/lee/data/macropic/newp/new_数据加强_整理版/'

while True:
    
    from multiprocessing.connection import Client
    from multiprocessing.connection import Listener

    data_augmentation.run(first_path, data_augmentation_save_path)
    run_tidy(data_augmentation_save_path, tidy_save_path)
    run_tensorflow(tidy_save_path)
    
    print '检测到{}个新图片'.format(num / 11)

    if num is not 0:
        c = Client('/usr/local/server%d.temp' % 99, authkey=b'lee123456')
        # 将信息传送给服务端
        c.send(['-r'])
        # 等待服务端处理结果
        ar = c.recv()