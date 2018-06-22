#coding=utf-8

from tools import file_tools

# 将要存入的文件
list_file_path = 'list.txt'
file = None

"""将所需文件按照label格式存入到文件中

Raises:
    ValueError -- file_path为空时发出
"""

label_text = {
    '网布': 0,
    '动物': 1,
    '格子': 2,
    '混合': 3,
    '几何': 4,
    '其它': 5,
    '人': 6,
    '素色': 7,
    '条纹': 8,
    '圆点': 9,
    '植物': 10
}

def make_label(file_path):
    if file_path is None:
        raise ValueError('Please Check File\'s Path Not Be None', file_path)
    
    id_val = file_tools.getFloderOfFile(file_path, 2)
    if id_val == '混':
        print 'Bad {} {}\n'.format(file_path, id_val)
    else:
        try:
            file.write('{} {}\n'.format(file_path, label_text[id_val]))
        except:
            print 'Bad {} {}\n'.format(file_path, id_val)
            # sys.exit()

def help():
    print '用法: -f [图片路径] [选项]... [选项]...'
    print ''
    print '将文件夹内的指定格式的图片文件进行制作成列表，Label为上一级文件夹名称(路径中最好不要带有空格)'
    print ''
    print '必选参数.'
    print '-f 指定图片所在文件夹路径'
    print ''
    print '可选参数.'
    print '-l 指定生成的文件路径 默认为list.txt'
    print '-t 指定图片格式，默认为jpg'
    print ''

if __name__ == '__main__':
    
    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], 'f:l:t:h')

    path = None
    check_file_format = 'jpg'

    
    for op, value in opts:
        if op == '-f':
           path = value 
        elif op == '-l':
            list_file_path = value
        elif op == '-t':
            check_file_format = value
        elif op == '-h':
            help()

    if path is None:
        help()
        print '必须使用-f指定图片路径'
    else:            
        file = open(list_file_path, 'w')

        file_tools.traverse_floder(path, make_label, check_file_format)
        
        file.close()