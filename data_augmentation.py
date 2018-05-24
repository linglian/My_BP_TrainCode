#coding=utf-8

from PIL import Image
import cv2
import numpy as np
import os
import sys
import getopt

# 数据增强区
is_rotate = False
is_center_crop = True
mode = 10
image_format = '.webp'

mainFold = None
toFold = None

"""创建文件夹（如果文件夹不存在的话）
"""

def check_fold(name):
    if not os.path.exists(name):
        os.mkdir(name)

def getCrop(image_filepath, is_rotate=False, is_center_crop=True, img_size=(224, 224), mode=10):
    import cv2
    from PIL import Image
    def get_image_rotate(img, angle=0):
        if isinstance(img, Image.Image) is False:
            raise ArithmeticError('img Must be PIL.Image.Image', img)
        w, h = img.size
        rotate35_img_true = img.rotate(angle, expand=True)
        rotate35_img_false = img.rotate(angle, expand=False)
        width, height = rotate35_img_false.size
        width2, height2 = rotate35_img_true.size
        new_width = width - width2 + width
        new_height = height - height2 + height
        left = int((width - new_width) / 2)
        top = int((height - new_height) / 2)
        right = left + new_width
        bottom = top + new_height
        result_img = rotate35_img_false.crop((left, top, right, bottom))
        return result_img
    if mode != 5 and mode != 10:
        raise ArithmeticError('Mode only has 5 or 10', mode)
    try:
        img = Image.fromarray(cv2.imread(image_filepath))
    except Exception:
        raise ValueError('image_filepath is Bad or None', image_filepath)
    w, h = img.size
    # 进行Central-Crop
    if is_center_crop:
        img = img.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
        w, h = img.size
    img_array = []
    img_array.append(img)
    if is_rotate:
        rotate_array = [-30, -20, -10, 10, 20, 30]
        for angle in rotate_array:
            img_array.append(get_image_rotate(img, angle=angle))
    # 获取Crop-5
    img_array.append( img.crop( (0, 0,
                                               int(w * 0.5), int(h * 0.5))))
    img_array.append( img.crop( (int(w * 0.5), 0,
                                               w, int(h * 0.5))))
    img_array.append( img.crop( (0, int(h * 0.5),
                                               int(w * 0.5), h)))
    img_array.append( img.crop( (int(w * 0.5), int(h * 0.5),
                                               w, h)))
    img_array.append( img.crop( (int(w * 0.25), int(h * 0.25),
                                               int(w * 0.75), int(h * 0.75))))
    # 进而获取Crop-10
    if mode == 10:
        for i in range(5):
            img_array.append(img_array[i].transpose(Image.FLIP_LEFT_RIGHT))
    result_array = []
    for im in img_array:
        result_array.append(cv2.resize(np.array(im), img_size))
    return result_array

"""数据增强
    im: 需要增强的图片
    temp_list: 需要存在哪个数组
"""

def getImage(img):
    import numpy as np
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def splite_img(imgfile):
    import random
    try:
        newname = imgfile.replace('.', '_{:03d}.'.format(0))
        newname = newname.replace(mainFold, toFold)
        if os.path.exists(newname[0: newname.find('.')] + '.jpg') is True:
            return

        # 打开图片
        img_array = getCrop(imgfile, is_rotate=is_rotate, is_center_crop=is_center_crop, mode=mode)
        # 将图片增强tilesPerImage份
        for idx, im in enumerate(img_array):
            newname = imgfile.replace('.', '_{:03d}.'.format(idx))
            newname = newname.replace(mainFold, toFold)
            cv2.imwrite(newname[0: newname.find('.')] + '.jpg', im)
    except Exception as msg:
        print('Bad Image: %s B %s ' % (imgfile, msg))
        return None
 
def start_splite(path, filePath, toPath):
    print('Create Fold %s ' % os.path.join(toPath, filePath));
    
    check_fold(os.path.join(toPath, filePath));
    folders = [folder for folder in os.listdir(
        os.path.join(path, filePath)) if os.path.isdir(os.path.join(path, filePath, folder))]

    folders2 = [folder for folder in os.listdir(
        os.path.join(path, filePath)) if folder.endswith(image_format)]

    print(folders2)

    for folder in folders2:
        # print('Splite Image %s ' % os.path.join(toPath, filePath, folder));
        splite_img(os.path.join(path, filePath, folder))

    for folder in folders:
        start_splite(os.path.join(path, filePath), folder, os.path.join(toPath, filePath))


def help():
    print '用法: -f [图片路径] -s 保存路径 [选项]... [选项]...'
    print ''
    print '将文件夹内的指定格式的图片文件进行数据加强(文件夹目录结构会完美复制)'
    print ''
    print '必选参数.'
    print '-f 指定图片所在文件夹路径'
    print '-s 指定处理完的储存的文件夹路径'
    print ''
    print '可选参数.'
    print '-r 是否进行旋转[-30, -20, -10, 10, 20, 30]'
    print '-c 是否进行CenterCrop(切掉边缘的20%，有些图片边缘有一些公共的图片，比如标志，来源，或者边框)'
    print '-m 5|10 指定Crop-5或者Crop-10'
    print '-t 指定图片格式，默认为webp'
    print ''

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], 'f:s:rcm:ht:')
    for op, value in opts:
        # 设置根目录路径
        if op == '-f':
            if value.find('.') != -1:
                raise ValueError('路径必须为绝对路径', value)
            mainFold = value
        elif op == '-s':
            toFold = value
            if value.find('.') != -1:
                raise ValueError('路径必须为绝对路径', value)
        elif op == '-r':
            is_rotate = True
        elif op == '-c':
            is_center_crop = False
        elif op == '-m':
            mode = int(value)
        elif op == '-t':
            image_format = value
        elif op == '-h':
            help()
    if mainFold is None:
        help()
        print '必选使用-f 指定需要数据加强的文件夹'
    elif toFold is None:
        help()
        print'必选使用-s 指定图片存储路径'
    else:
        start_splite(mainFold, '', toFold);