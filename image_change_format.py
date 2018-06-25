#coding=utf-8
import cv2
import sys
import os

from tools import file_tools
from PIL import Image
import imutils
import time

base_path = None
save_path = None
na_format = '.webp'
af_format = '.jpg'
is_change_size = True

image_sum = 0
now_number = 0
last_number = 0
last_time = time.time()

def change_format(image_path):
    global image_sum, now_number, last_time, last_number
    nature_image_path = image_path
    image_path = image_path.replace(base_path, save_path)
    image_path = image_path.replace(na_format, af_format)
    if os.path.exists(image_path) is False:
        file_tools.check_fold(file_tools.getFloderOfFileJustPath(image_path))
        img = cv2.imread(nature_image_path)
        if img is not None:
            img = Image.fromarray(imutils.opencv2matplotlib(img))
            img.save(image_path)
        else:
            if os.path.exists(nature_image_path) is False:
                print '无法读取webp文件(文件不存在)...%s' % nature_image_path
            else:
                print '无法读取webp文件...%s' % nature_image_path
    now_number += 1
    if now_number % 100 == 0:
        print '正在进行转换...%d/%d(%.2f/sec)' % (now_number, image_sum, (float(now_number - last_number) / (time.time() - last_time)))
        last_number = now_number
        last_time = time.time()

def get_sum(image_path):
    global image_sum
    image_sum += 1

def help():
    print '用法: -f [图片路径] -s [保存路径] [选项]... [选项]...'
    print ''
    print '将文件夹内的指定格式的图片文件进行格式转换'
    print ''
    print '必选参数.'
    print '-f 指定图片所在文件夹路径'
    print '-s 指定图片保存路径'
    print ''
    print '可选参数.'
    print '-t 指定原图片格式，默认为.webp'
    print '-a 指定转换后的图片格式，默认为.jpg'
    print ''

if __name__ == '__main__':
    
    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], 'f:s:t:a:h:')

    path = None
    check_file_format = 'jpg'
    
    for op, value in opts:
        if op == '-f':
            base_path = value 
        elif op == '-s':
            save_path = value
        elif op == '-t':
            na_format = value
        elif op == '-a':
            af_format = value
        elif op == '-h':
            help()

    if base_path is None:
        help()
        print '必须使用-f指定图片路径'
    elif save_path is None:
        help()
        print '必须使用-s指定图片保存路径'
    else:
        base_path = file_tools.check_folder_name(base_path)
        save_path = file_tools.check_folder_name(save_path)
        file_tools.traverse_floder(base_folder=base_path, check_file_format=na_format, dothing_func=get_sum)
        file_tools.traverse_floder(base_folder=base_path, check_file_format=na_format, dothing_func=change_format)