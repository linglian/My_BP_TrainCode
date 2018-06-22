#coding=utf-8

import sys
import os
import cv2
from tools import file_tools

file_path = None
save_floder_path = None
file_format = '.jpg'

def let_do_it(file_path):
    file_name = file_tools.getFileName(file_path)
    class_name = file_tools.getFloderOfFile(file_path, 1)
    file_list = []
    if os.path.exists(os.path.join(save_floder_path, class_name)):
        file_list = os.listdir(os.path.join(save_floder_path, class_name))
    file_tools.copy_file(file_path, os.path.join(save_floder_path, class_name, file_name + file_format))
    print os.path.join(save_floder_path, class_name, file_name)
    
def help():
    print '用法: -f [图片路径] [选项]... [选项]...'
    print ''
    print '将文件夹内的指定格式的图片文件进行整理到一个文件下，在文件目录下以Label为上一级文件，Label为上一级文件夹名称(路径中最好不要带有空格)'
    print ''
    print '必选参数.'
    print '-f 指定图片所在文件夹路径'
    print '-s 存储路径'
    print ''
    print '可选参数.'
    print '-t 指定图片格式，默认为.jpg'
    print ''

if __name__ == '__main__':
    
    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], 'f:s:t:h')

    for op, value in opts:
        if op == '-f':
            file_path = value
        elif op == '-s':
            save_floder_path = value
        elif op == '-t':
            file_format = value
        elif op == '-h':
            help()
    
    if file_path is None:
        help()
        print '必须使用 -f 指定图片路径'
    elif save_floder_path is None:
        help()
        print '必须使用 -s 指定存储路径'
    else:
        file_tools.traverse_floder(file_path, let_do_it, check_file_format=file_format)

