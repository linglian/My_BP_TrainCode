#coding=utf-8

import random

def make_train_test_list(list_file_path, train_file_path='train_list.txt', test_file_path='test_list.txt', train_radio=0.95):
    train_file = open(train_file_path, 'w')
    test_file = open(test_file_path, 'w')
    list_file = open(list_file_path, 'r')

    files_list = list_file.readlines()

    list_file.close()

    # 将数据打乱
    random.shuffle(files_list)

    # 获取训练数据数量
    train_size = int(len(files_list) * train_radio)

    # 写入训练数据
    for file in files_list[:train_size]:
        train_file.write(file)
    
    train_file.close()

    # 写入验证数据
    for file in files_list[train_size:]:
        test_file.write(file)
    
    test_file.close()
        
def help():
    print '用法: -l [列表文件路径] [选项]... [选项]...'
    print ''
    print '将列表中的图片和Label分为Train和Test二个部分'
    print ''
    print '必选参数.'
    print '-l 指定列表文件，由 make_label.py 生成'
    print ''
    print '可选参数.'
    print '-t 指定生成的train_list的路径，默认为train_list.txt'
    print '-r 指定生成的test_list的路径，默认为test_list.tfrecord'
    print '-b 指定生成比例，默认为0.95'
    print ''

if __name__ == '__main__':
    
    import sys
    import getopt

    train_file_path = 'train_list.txt'
    test_file_path = 'test_list.txt'
    list_file_path = None
    train_radio = 0.95

    opts, args = getopt.getopt(sys.argv[1:], 'r:t:l:b:h')

    for op, value in opts:
        if op == '-r':
            train_file_path = value
        elif op == '-t':
            test_file_path = value
        elif op == '-l':
            list_file_path = value
        elif op == '-b':
            train_radio = float(value)
        elif op == '-h':
            help()
    
    if list_file_path is None:
        help()
        print '必须使用 -l 指定列表文件'
    else:
        make_train_test_list(list_file_path, train_file_path, test_file_path, train_radio)
    

