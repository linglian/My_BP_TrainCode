#coding=utf-8   
from multiprocessing.connection import Client
from multiprocessing.connection import Listener

import numpy as np

resp = {
    'info': '',
    'body': {}}

img = '/media/lee/data/macropic/宏观分类自采/格子/1201/DSC03471.JPG'

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

print resp
