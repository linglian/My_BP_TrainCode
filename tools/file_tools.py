#coding=utf-8

"""移动文件
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def move_file(srcfile,dstfile):
    import shutil
    import os
    if not os.path.isfile(srcfile):
        raise ValueError('srcfile not exist!', srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        
"""复制文件
"""
def copy_file(srcfile,dstfile):
    import shutil
    import os
    if not os.path.isfile(srcfile):
        raise ValueError('srcfile not exist!', srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件


"""获取文件的父路径
path: 完整的文件路径
index: 返回第几级父文件夹

Returns:
    str -- 返回父文件夹文字
"""

def getFloderOfFile(path, index=1):
    return path.split('/')[-1 - index]

"""获取文件名
path: 完整的文件路径

Returns:
    str -- 返回文件名
"""

def getFileName(path):
    try:
        return getFloderOfFile(path, 0);
    except Exception as msg:
        raise ValueError('Bad Path', path)

"""获取文件的路径(不包含文件名)
path: 完整的文件路径

Returns:
    str -- 返回文件的路径(不包含文件名)
"""

def getFloderOfFileJustPath(path):
    return path[:path.index(getFileName(path))]

"""遍历所有文件夹，并且对每个符合格式的文件运行函数
base_folder: 根路径
dothing_func: 执行函数
check_file_format: 文件格式
"""

def traverse_floder(base_folder, dothing_func, check_file_format='jpg', is_log=True):
    import os

    # 获取base_folder目录下的所有文件
    floders_list = [folder for folder in
        os.listdir(base_folder) if
        os.path.isdir(os.path.join(base_folder, folder))]
    
    # 获取base_folder目录下的check_file_format后缀的文件
    files_list = [file for file in
        os.listdir(base_folder) if
        check_file_format is None or file.endswith(check_file_format)]

    # 执行给定函数操作
    for file in files_list:
        dothing_func(os.path.join(base_folder, file))

    # 递归遍历所有文件夹
    for floder in floders_list:
        traverse_floder(os.path.join(base_folder, floder), dothing_func, check_file_format)


"""创建文件夹（如果文件夹不存在的话）
"""

def check_fold(name):
    import os
    if name.endswith('/'):
        name = name[:-1]
    #print name
    if not os.path.exists(name):
        if not os.path.exists(getFloderOfFileJustPath(name)):
            check_fold(getFloderOfFileJustPath(name))
        os.mkdir(name)

"""设置路径为标准文件路径
"""

def check_folder_name(name):
    if not name.endswith('/'):
        name = name + '/' 
    return name 