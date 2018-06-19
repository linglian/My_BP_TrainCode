
from multiprocessing.connection import Client
from multiprocessing.connection import Listener

resp = {
    'info': '',
    'body': {}}

img = '/media/lee/data/macropic/宏观分类自采/格子/1201/DSC03471.JPG'
c = Client('/usr/local/server%d.temp' % 99, authkey=b'lee123456')
# 将信息传送给服务端
c.send(['-f', img])
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
        
class_name_list = ar[t_idx]
file_path_list = ar[t_idx + 1]
feature_list = ar[t_idx + 2]
featrue = ar[t_idx + 3]
resp['body']['class_name_list'] = class_name_list
resp['body']['file_path_list'] = file_path_list
resp['body']['feature_list'] = feature_list
resp['body']['featrue'] = featrue
print resp
