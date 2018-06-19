#coding=utf-8   
from multiprocessing.connection import Client
from multiprocessing.connection import Listener

import numpy as np

if __name__ == '__main__':
    import getopt
    import sys
    img = None
    opts, args= getopt.getopt(sys.argv[1:], 'f:')
    for op, value in opts:
        if op == '-f':
            img = value

    if img is None:
        print '必须使用-f指定图片路径'
    else:
        resp = {
            'info': '',
            'body': {}
            }

        c = Client('/usr/local/server%d.temp' % 99, authkey=b'lee123456')
        # 将信息传送给服务端
        c.send(['-f', img, '-k', 50])
        # 等待服务端处理结果
        ar = c.recv()
        is_Ok = False
        t_idx = 0
        for idx, i in enumerate(ar):
            if is_Ok:
                t_idx = idx
                break
            elif i == 'Next':
                is_Ok = True

        pre_response = ar[t_idx]

        resp['body']['response'] = pre_response

        pre_log_str = ''

        for idx, i in enumerate(pre_response):
            pre_log_str = pre_log_str + '\n' + 'No.%d id: %s file_path: %s cos_score: %.2f%%' % ((idx + 1), i[0], i[1], i[2])

        print pre_log_str