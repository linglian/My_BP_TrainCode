# coding=utf-8
from django.http import HttpResponse
from django.shortcuts import render
import json
import cv2
import numpy as np

my_id = 99

def uploadImg(request):
    content = {
        'imgs': [],
    }
    if request.method == 'POST':
        f=request.FILES['img']
        with open('static/image/temp.jpg', 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        img = cv2.imread('static/image/temp.jpg')

        img = cv2.resize(np.array(img), (224, 224))

        cv2.imwrite('static/image/temp.jpg', img)

        result_list = [['8156', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/8156/DSC03356_004.jpg.jpg', 0.58208764, 50.43857089347062], ['14680', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/14680/DSC04230_010.jpg.jpg', 0.58099395, 51.25568434470864], ['3609', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/3609/DSC00729_001.jpg.jpg', 0.57592934, 51.11042888090711], ['14682', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/14682/DSC04217_003.jpg.jpg', 0.5521012, 61.5643832863206], ['3686', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/3686/DSC02410_007.jpg.jpg', 0.53848475, 51.31360428316892], ['11717', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/11717/DSC06235_006.jpg.jpg', 0.5361875, 50.929637257442856], ['4518', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/4518/DSC02812_006.jpg.jpg', 0.5316246, 52.029720080687774], ['9951', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/9951/DSC05889_008.jpg.jpg', 0.5168723, 62.0271868037661], ['11282', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/11282/DSC00993_008.jpg.jpg', 0.51229084, 55.1746412992437], ['14237', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/14237/DSC01956_003.jpg.jpg', 0.51204544, 54.96618432923426], ['408', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/408/DSC06301_004.jpg.jpg', 0.5106615, 52.025112783237965], ['4402', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/4402/DSC06494_010.jpg.jpg', 0.5064689, 55.7835028003309], ['3608', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/3608/DSC00734_004.jpg.jpg', 0.5061582, 55.58985385003185], ['11734', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/11734/DSC06071_000.jpg.jpg', 0.5052108, 52.03575504385509], ['1419', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/1419/DSC01657_005.jpg.jpg', 0.50392884, 55.23767258542725], ['350', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/350/DSC06613_006.jpg.jpg', 0.50158376, 55.423559922523225], ['344', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/344/DSC06282 (2)_006.jpg.jpg', 0.5004241, 59.1730091260855], ['15567', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/15567/DSC04148_000.jpg.jpg', 0.49882323, 54.73350102914844], ['15518', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/15518/DSC04839_005.jpg.jpg', 0.49863857, 58.144111485526416], ['2815', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/2815/DSC02164_006.jpg.jpg', 0.49495333, 51.20426656400296], ['3585', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/3585/DSC06747_005.jpg.jpg', 0.49099973, 52.80337367470275], ['15538', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/15538/DSC04314_010.jpg.jpg', 0.48977968, 59.797982809685436], ['15544', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/15544/DSC04281_001.jpg.jpg', 0.48943734, 55.778887609830505], ['1887', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/1887/DSC04657_001.jpg.jpg', 0.4892835, 56.85685067588267], ['1741', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/1741/IMG_4165_000.jpg.jpg', 0.48624516, 53.37004471204811], ['4328', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/4328/DSC05852_000.jpg.jpg', 0.48456773, 57.26868773138316], ['2829', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/2829/DSC06172_000.jpg.jpg', 0.48406315, 51.22049501858206], ['15582', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/15582/DSC04057_008.jpg.jpg', 0.48382664, 57.2835468132801], ['3555', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/3555/DSC09951_004.jpg.jpg', 0.4830583, 55.34372663985789], ['710', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/710/IMG_4824_010.jpg.jpg', 0.48300642, 56.0515250243998], ['419', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/419/DSC06568_000.jpg.jpg', 0.47950545, 53.50245113405047], ['14686', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/14686/DSC04187_008.jpg.jpg', 0.47896156, 55.73280091441133], ['2830', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/2830/DSC06169_000.jpg.jpg', 0.478447, 50.16538894231163], ['2840', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/2840/DSC06079_004.jpg.jpg', 0.47466227, 53.25906165179439], ['10986', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/10986/DSC02634_005.jpg.jpg', 0.47373348, 61.05096803033399], ['4347', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/4347/DSC05696_007.jpg.jpg', 0.47238806, 53.51640256749811], ['4287', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/4287/DSC05008_009.jpg.jpg', 0.47023708, 57.17674911969772], ['2828', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/2828/DSC06180_007.jpg.jpg', 0.46687064, 51.21013842212678], ['4736', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/4736/DSC03464_006.jpg.jpg', 0.46596882, 52.58884026444999], ['6399', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/6399/DSC02107_008.jpg.jpg', 0.46547985, 58.33808485330444], ['15539', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/15539/DSC04309_009.jpg.jpg', 0.46483818, 54.28099387080212], ['1621', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/1621/DSC01485_003.jpg.jpg', 0.46483427, 55.46462996257647], ['11779', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/11779/DSC04814_005.jpg.jpg', 0.46438104, 56.158328963144406], ['892', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/892/DSC01414_003.jpg.jpg', 0.46332833, 51.837919942179674], ['3673', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/3673/DSC02510_001.jpg.jpg', 0.4631903, 56.69494994920799], ['838', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/838/DSC00864_008.jpg.jpg', 0.46183977, 58.57462151239557], ['1960', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/1960/IMG_3622_007.jpg.jpg', 0.4616541, 56.4354989658871], ['15566', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/15566/DSC04153_003.jpg.jpg', 0.4616524, 55.52511911758496], ['4715', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/4715/DSC03561_000.jpg.jpg', 0.46154907, 54.73802923142161], ['15534', '/media/lee/data/macropic/newp/new_\xe6\x95\xb0\xe6\x8d\xae\xe5\x8a\xa0\xe5\xbc\xba\xe7\x89\x88_\xe6\x95\xb4\xe7\x90\x86\xe7\x89\x88/15534/DSC04337_010.jpg.jpg', 0.4600142, 58.684083026669654]]
        
        for i in result_list:
            file_name = i[1]
            i[1] = '/medias/' + file_name.split('/')[-2] + '/' + file_name.split('/')[-1].split('_')[0] + '_000' + file_name.split('/')[-1].split('_')[1][3:]
            i[2] = '%.2f%%' % i[2]
        content = {
            'imgs': result_list,
        }

    return render(request, 'uploading.html', content)

def pre_cloth(request):
    from multiprocessing.connection import Client
    from multiprocessing.connection import Listener

    resp = {
        'info': '',
        'body': {}}

    if request.method == 'GET':
        img = request.GET['img']
        c = Client('/usr/local/server%d.temp' % my_id, authkey=b'lee123456')
        # 将信息传送给服务端
        c.send(['-f', img])
        # 等待服务端处理结果
        ar = c.recv()
        is_Ok = False
        t_idx = 0
        for idx, i in enumerate(ar):
            if i == 'Next':
                is_Ok = True
            elif is_Ok:
                t_idx = idx
                break
        class_name_list = ar[t_idx]
        file_path_list = ar[t_idx + 1]
        feature_list = ar[t_idx + 2]
        featrue = ar[t_idx + 3]
        resp['body']['class_name_list'] = class_name_list
        resp['body']['file_path_list'] = file_path_list
        resp['body']['feature_list'] = feature_list
        resp['body']['featrue'] = featrue
    return HttpResponse(json.dumps(resp), content_type="application/json")

def get(request):
    resp = {
        'price': {
            'sample': 22,
            'cloth': 20
        },
        'colors': [
            {
                'name': '杏色',
                'cover': '/assets/images/sys.png'
            }, {
                'name': '红色',
                'cover': '/assets/images/yjfk.png'
            }, {
                'name': '分色',
                'cover': '/assets/images/zbzz.png'
            }, {
                'name': '皇色',
                'cover': '/assets/images/zhsz.png'
            }
        ]
    }
    return HttpResponse(json.dumps(resp), content_type="application/json")


def getButton(request):
    '''
    调用方式: wx.request
    url: https://by.edenhe.com/api/shelf/goods/shuaixuan/
    method: GET
    参数: 无
    返回值:
    {
        msg,
        data,
        error,
    }
    data:
    格式说明:
        一共三级，实现筛选的三级联动。
        第一级: 返回第一级类型名和第二级数组
        第二级: 返回第二级类型名和第三级数组
        第三级：返回类型名
    '''
    resp = {
        'msg': '',
        'data': [
            {
                'name': '材质',
                'userLevel': True,
                'array': [
                    {
                        'name': '棉类',
                        'array': [
                            '纱卡',
                            '平布',
                            '净色布'
                        ]
                    },
                    {
                        'name': '麻类',
                        'array': [
                            '全棉布',
                            '其他',
                            '绵竹布'
                        ]
                    }
                ]
            },
            {
                'name': '图案',
                'userLevel': False,
                'array': [
                    {
                        'name': '人工',
                        'array': [
                            '熊猫',
                            '猫',
                            '狗'
                        ]
                    },
                    {
                        'name': '机器',
                        'array': [
                            '机器人',
                            '猪',
                            '羊'
                        ]
                    }
                ]
            }
        ],
        'error': ''
    }
    '''
    key说明:
        name: 当前级别的名字
        array: 当前级别下一级的数组
        userLevel: 用户权限是否可观看。
    '''
    return HttpResponse(json.dumps(resp), content_type="application/json")


def list(request):
    resp = {
        'data': [
            {
                'clothName': '杏色',
                'cover': '/assets/images/sys.png',
                'price': {
                    'sample': 21,
                    "cloth": 23,
                    "unit": "m"
                },
                "clothID": "20",
                "thumb": "/assets/images/buliao.jpg"
            },
            {
                'clothName': '黄色',
                'cover': '/assets/images/sys.png',
                'price': {
                    'sample': 21,
                    "cloth": 23,
                    "unit": "m"
                },
                "clothID": "20",
                "thumb": "/assets/images/buliao.jpg"
            },
            {
                'clothName': '金色',
                'cover': '/assets/images/sys.png',
                'price': {
                    'sample': 21,
                    "cloth": 23,
                    "unit": "m"
                },
                "clothID": "20",
                "thumb": "/assets/images/buliao.jpg"
            },
            {
                'clothName': '白色',
                'cover': '/assets/images/sys.png',
                'price': {
                    'sample': 21,
                    "cloth": 23,
                    "unit": "m"
                },
                "clothID": "20",
                "thumb": "/assets/images/buliao.jpg"
            }
        ]
    }
    return HttpResponse(json.dumps(resp), content_type="application/json")


def getIndex(request):
    resp = {
        'msg': '',
        'error': '',
        'data': {
            'tuijian': [
                {
                    'cloth_id': '1',
                    'src': '/assets/images/buliao.jpg'
                },
                {
                    'cloth_id': '11',
                    'src': '/assets/images/buliao.jpg'
                },
                {
                    'cloth_id': '111',
                    'src': '/assets/images/buliao.jpg'
                },
                {
                    'cloth_id': '1111',
                    'src': '/assets/images/buliao.jpg'
                },
            ],
            'jingxuan': {
                'cloth_id': '4',
                'src': '/assets/images/buliao.jpg'
            },
            'zuixinshangjia': {
                'cloth_id': '5',
                'src': '/assets/images/buliao.jpg'
            },
            'dangjiqushi': {
                'cloth_id': '6',
                'src': '/assets/images/buliao.jpg'
            },
            'shishangremai': {
                'cloth_id': '7',
                'src': '/assets/images/buliao.jpg'
            }
        }
    }
    return HttpResponse(json.dumps(resp), content_type="application/json")


"""获取订单详情
    接口说明:
        根据订单编号，返回订单详细信息
    请求方式:
        GET
    传入参数:
        无
    响应内容:
        msg: 提示内容,
        data: { 响应数据
            id: 订单编号,
            status: 订单状态（未付款、已付款、正在付款）,
            people: { 收货信息
                name: 收货人姓名,
                address: 收货人地址,
                phone: 收货人电话
            },
            all_items: [ 订单购买的商品列表
                    { 商品信息
                    shop_id: 商铺编号,
                    shop_name: 商铺名称,
                    items: [
                        { 购买的商品列表
                            id: 商品编号,
                            src: 商品图片,
                            num: 商品数量,
                            price: 商品价格,
                            unit: 购买单位,
                            name: 商品名称,
                            type: 商品类型,
                            color: 商品颜色,
                            guayang_huowei: 挂样货位,
                            seka_huowei: 色卡货位
                        }
                    ]
                ]
            },
            item_sum: 商品总金额,
            freight: 运费,
            discounts: 优惠,
            last_money: 实际付款,
            time: 下单时间(YYYY-MM-DD hh:mm:ss)
        },
        error: 错误提示
    相应实例:
        resp = {
            'msg': '',
            'data': {
                'id': 123456,
                'status': '未付款',
                'people': {
                    'name': '收货人姓名',
                    'address': '收货人地址',
                    'phone': '收货人电话'
                },
                'all_items': [
                    {
                        'shop_id': 1,
                        'shop_name': '商铺名称',
                        'items': [
                            {
                                'id': 1,
                                'src': '../../assets/images/buliao.jpg',
                                'num': 1,
                                'price': 1,
                                'unit': '米',
                                'name': '商品名称',
                                'type': '商品类型',
                                'color': '商品颜色',
                                'guayang_huowei': '挂样货位',
                                'seka_huowei': '色卡货位'
                            },
                            {
                                'id': 2,
                                'src': '../../assets/images/buliao.jpg',
                                'num': 2,
                                'price': 2,
                                'unit': '米',
                                'name': '商品名称',
                                'type': '商品类型',
                                'color': '商品颜色',
                                'guayang_huowei': '挂样货位',
                                'seka_huowei': '色卡货位'
                            }
                        ]
                    },
                    {
                        'shop_id': 2,
                        'shop_name': '商铺名称',
                        'items': [
                            {
                                'id': 3,
                                'src': '../../assets/images/buliao.jpg',
                                'num': 3,
                                'price': 3,
                                'unit': '米',
                                'name': '商品名称',
                                'type': '商品类型',
                                'color': '商品颜色',
                                'guayang_huowei': '挂样货位',
                                'seka_huowei': '色卡货位'
                            },
                            {
                                'id': 4,
                                'src': '../../assets/images/buliao.jpg',
                                'num': 4,
                                'price': 4,
                                'unit': '米',
                                'name': '商品名称',
                                'type': '商品类型',
                                'color': '商品颜色',
                                'guayang_huowei': '挂样货位',
                                'seka_huowei': '色卡货位'
                            }
                        ]
                    }
                ],
                'item_sum': 30,
                'freight': 0,
                'discounts': 0,
                'last_money': 30,
                'time': '2017-01-02 12:34:56'
            },
            'error': ''
        }
"""


def dingdan(request):
    resp = {
        'msg': '',
        'data': {
            'id': 123456,
            'status': '未付款',
            'people': {
                'name': '陌生人',
                'address': '地球',
                'phone': '12312341234'
            },
            'all_items': [
                {
                    'shop_id': 1,
                    'shop_name': '港口旺铺',
                    'items': [
                        {
                            'id': 1,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '小帆布',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                        {
                            'id': 2,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 2,
                            'price': 2,
                            'unit': '米',
                            'name': '大帆布',
                            'type': '大货',
                            'color': '蓝色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        }
                    ]
                },
                {
                    'shop_id': 2,
                    'shop_name': '小杂货',
                    'items': [
                        {
                            'id': 3,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 3,
                            'price': 3,
                            'unit': '米',
                            'name': '羊绒混搭',
                            'type': '大货',
                            'color': '杏色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                        {
                            'id': 4,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 4,
                            'price': 4,
                            'unit': '米',
                            'name': '牛绒混搭',
                            'type': '大货',
                            'color': '黑色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        }
                    ]
                }
            ],
            'item_sum': 30,
            'freight': 0,
            'discounts': 0,
            'last_money': 30,
            'time': '2017-01-02 12:34:56'
        },
        'error': ''
    }
    return HttpResponse(json.dumps(resp), content_type="application/json")


"""获取用户购物车
    接口说明:
        根据用户的标识来获取用户当前购物车
    请求方式:
        GET
    传入参数:
        无
    响应内容:
        msg: 提示的信息,
        data: {
            'all_items': [
                {
                    'shop_id': 1,
                    'shop_name': '商铺名称',
                    'items': [
                        {
                            'id': 1,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '商品名称',
                            'type': '商品类型',
                            'color': '商品颜色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                        {
                            'id': 2,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 2,
                            'price': 2,
                            'unit': '米',
                            'name': '商品名称',
                            'type': '商品类型',
                            'color': '商品颜色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        }
                    ]
                },
                {
                    'shop_id': 2,
                    'shop_name': '商铺名称',
                    'items': [
                        {
                            'id': 3,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 3,
                            'price': 3,
                            'unit': '米',
                            'name': '商品名称',
                            'type': '商品类型',
                            'color': '商品颜色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                        {
                            'id': 4,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 4,
                            'price': 4,
                            'unit': '米',
                            'name': '商品名称',
                            'type': '商品类型',
                            'color': '商品颜色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        }
                    ]
                }
            ]
        },
        error: 错误提示信息
"""


def gwc(request):
    resp = {
        'msg': '',
        'data': {
            'all_items': [
                {
                    'shop_id': 1,
                    'shop_name': '港口旺铺',
                    'items': [
                        {
                            'id': 1,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '小帆布',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                        {
                            'id': 2,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 2,
                            'price': 2,
                            'unit': '米',
                            'name': '大帆布',
                            'type': '大货',
                            'color': '蓝色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        }
                    ]
                },
                {
                    'shop_id': 2,
                    'shop_name': '小杂货',
                    'items': [
                        {
                            'id': 3,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 3,
                            'price': 3,
                            'unit': '米',
                            'name': '羊绒混搭',
                            'type': '大货',
                            'color': '杏色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                        {
                            'id': 4,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 4,
                            'price': 4,
                            'unit': '米',
                            'name': '牛绒混搭',
                            'type': '大货',
                            'color': '黑色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        }
                    ]
                }
            ]
        },
        'error': ''
    }
    return HttpResponse(json.dumps(resp), content_type="application/json")


"""获取用户自定义设计单列表，以及设计单涉及的布料
    接口说明:
        根据用户的标识来获取用户自定义的设计单列表，以及设计单涉及的布料
    请求方式:
        GET
    传入参数:
        无
    响应内容:
        msg: 提示的信息,
        data: {
            'sjd_list': [  
                {
                    'name': '棉麻',
                    'cloth_list': [
                        {
                            'id': 1,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '小帆布',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                        {
                            'id': 2,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '大帆布',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                    ]
                },  
                {
                    'name': '格子',
                    'cloth_list': [
                        {
                            'id': 1,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '小格子',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                        {
                            'id': 2,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '大格子',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位'
                        },
                    ]
                }
            ]
        },
        error: 错误提示信息
"""


def sjd_get(request):
    resp = {
        'msg': '提示的信息',
        'data': {
            'list': [
                {
                    'name': '棉麻',
                    'cloth_list': [
                        {
                            'id': 1,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '小帆布',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位',
                            'is_pay': False
                        },
                        {
                            'id': 2,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '大帆布',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位',
                            'is_pay': True
                        },
                    ]
                },
                {
                    'name': '格子',
                    'cloth_list': [
                        {
                            'id': 3,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '小格子',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位',
                            'is_pay': True
                        },
                        {
                            'id': 4,
                            'src': '../../assets/images/buliao.jpg',
                            'num': 1,
                            'price': 1,
                            'unit': '米',
                            'name': '大格子',
                            'type': '大货',
                            'color': '橘色',
                            'guayang_huowei': '挂样货位',
                            'seka_huowei': '色卡货位',
                            'is_pay': False
                        },
                    ]
                },
            ]
        },
        'error': '错误提示信息'
    }
    return HttpResponse(json.dumps(resp), content_type="application/json")


"""删除指定设计单下的指定id的布料
    接口说明:
        删除指定设计单下的指定id的布料
    请求方式:
        GET
    传入参数:
        tyoe_name: 设计单名称
        cloth_id: 布料id
    响应内容:
        msg: 提示的信息,
        data: {
            is_success: 是否删除成功(True or False),
            text: 删除失败的提示信息
        }
        error: 错误提示信息
"""


def sjd_remove(request):
    resp = {
        'msg': '',
        'error': '',
        'data': {
            'is_success': False,
            'text': '现在就不让你删，你能怎么滴呢'
        }
    }
    if request.method == 'GET':
        type_name = request.GET['type_name']
        cloth_id = request.GET['cloth_id']
        print(type_name)
        print(cloth_id)
    return HttpResponse(json.dumps(resp), content_type="application/json")


"""将面料根据id列表添加到指定设计单中
    接口说明:
        将面料根据id列表添加到指定设计单中
    请求方式:
        GET
    传入参数:
        add_type_name: 设计单名称
        addList: 布料id列表
    响应内容:
        msg: 提示的信息,
        data: {
            is_success: 是否添加成功(True or False),
            text: 添加失败的提示信息
        }
        error: 错误提示信息
"""


def sjd_add_cloth(request):
    resp = {
        'msg': '',
        'error': '',
        'data': {
            'is_success': True,
            'text': '添加面料失败'
        }
    }
    if request.method == 'GET':
        addList = request.GET['addList']
        add_type_name = request.GET['add_type_name']
        print(addList)
        print(add_type_name)
    return HttpResponse(json.dumps(resp), content_type="application/json")


"""获取所有面料信息
    接口说明:
        将面料根据id列表添加到指定设计单中
    请求方式:
        GET
    传入参数:
        无
    响应内容:
        msg: 提示的信息,
        data: {
            'num': [ 开头字母列表（哪些有就返回哪些）
                'a',
                'b',
            ],
            list: {
                'a': [ 布料名字开头字母为a的布料列表
                    {
                        id: 商品编号,
                        src: 商品图片,
                        num: 商品数量,
                        price: 商品价格,
                        unit: 购买单位,
                        name: 商品名称,
                        type: 商品类型,
                        color: 商品颜色,
                        guayang_huowei: 挂样货位,
                        seka_huowei: 色卡货位
                        is_pay: 供应商是否交费
                    },
                    {
                        id: 商品编号,
                        src: 商品图片,
                        num: 商品数量,
                        price: 商品价格,
                        unit: 购买单位,
                        name: 商品名称,
                        type: 商品类型,
                        color: 商品颜色,
                        guayang_huowei: 挂样货位,
                        seka_huowei: 色卡货位
                        is_pay: 供应商是否交费
                    },
                ]，
                'b': [ 布料名字开头字母为b的布料列表
                    {
                        id: 商品编号,
                        src: 商品图片,
                        num: 商品数量,
                        price: 商品价格,
                        unit: 购买单位,
                        name: 商品名称,
                        type: 商品类型,
                        color: 商品颜色,
                        guayang_huowei: 挂样货位,
                        seka_huowei: 色卡货位
                        is_pay: 供应商是否交费
                    },
                    {
                        id: 商品编号,
                        src: 商品图片,
                        num: 商品数量,
                        price: 商品价格,
                        unit: 购买单位,
                        name: 商品名称,
                        type: 商品类型,
                        color: 商品颜色,
                        guayang_huowei: 挂样货位,
                        seka_huowei: 色卡货位
                        is_pay: 供应商是否交费
                    },
                ]

            }
        }
        error: 错误提示信息
"""


def get_all_cloth(request):
    resp = {
        'msg': '',
        'error': '',
        'data': {
            'num': [
                'a',
                'b',
                'c',
            ],
            'list': {
                'a': [
                    {
                        'id': 1,
                        'src': '../../assets/images/buliao.jpg',
                        'num': 1,
                        'price': 1,
                        'unit': '米',
                        'name': 'a小帆布',
                        'type': '大货',
                        'color': '橘色',
                        'guayang_huowei': '挂样货位',
                        'seka_huowei': '色卡货位',
                        'is_pay': False
                    },
                    {
                        'id': 2,
                        'src': '../../assets/images/buliao.jpg',
                        'num': 1,
                        'price': 1,
                        'unit': '米',
                        'name': 'a大帆布',
                        'type': '大货',
                        'color': '橘色',
                        'guayang_huowei': '挂样货位',
                        'seka_huowei': '色卡货位',
                        'is_pay': True
                    },
                ],
                'b': [
                    {
                        'id': 3,
                        'src': '../../assets/images/buliao.jpg',
                        'num': 1,
                        'price': 1,
                        'unit': '米',
                        'name': 'b大帆布',
                        'type': '大货',
                        'color': '橘色',
                        'guayang_huowei': '挂样货位',
                        'seka_huowei': '色卡货位',
                        'is_pay': True
                    },
                    {
                        'id': 4,
                        'src': '../../assets/images/buliao.jpg',
                        'num': 1,
                        'price': 1,
                        'unit': '米',
                        'name': 'b大帆布',
                        'type': '大货',
                        'color': '橘色',
                        'guayang_huowei': '挂样货位',
                        'seka_huowei': '色卡货位',
                        'is_pay': True
                    },
                    {
                        'id': 5,
                        'src': '../../assets/images/buliao.jpg',
                        'num': 1,
                        'price': 1,
                        'unit': '米',
                        'name': 'b大帆布',
                        'type': '大货',
                        'color': '橘色',
                        'guayang_huowei': '挂样货位',
                        'seka_huowei': '色卡货位',
                        'is_pay': True
                    },
                    {
                        'id': 6,
                        'src': '../../assets/images/buliao.jpg',
                        'num': 1,
                        'price': 1,
                        'unit': '米',
                        'name': 'b大帆布',
                        'type': '大货',
                        'color': '橘色',
                        'guayang_huowei': '挂样货位',
                        'seka_huowei': '色卡货位',
                        'is_pay': True
                    }
                ],
                'c': [
                    {
                        'id': 7,
                        'src': '../../assets/images/buliao.jpg',
                        'num': 1,
                        'price': 1,
                        'unit': '米',
                        'name': 'c大帆布',
                        'type': '大货',
                        'color': '橘色',
                        'guayang_huowei': '挂样货位',
                        'seka_huowei': '色卡货位',
                        'is_pay': True
                    },
                ],

            }
        }
    }
    return HttpResponse(json.dumps(resp), content_type="application/json")

"""保存设计单
    接口说明:
        将设计单进行保存
    请求方式:
        GET
    传入参数:
        type_name: 设计单名称
        new_type_name: 新设计单名称
        cloth_list: 布料id列表
    响应内容:
        msg: 提示的信息,
        data: {
            is_success: 是否保存成功(True or False),
            text: 保存失败的提示信息
        }
        error: 错误提示信息
"""

def save_sjd(request):
    resp = {
        'msg': '',
        'error': '',
        'data': {
            'is_success': True,
            'text': '添加设计单失败'
        }
    }
    if request.method == 'GET':
        type_name = request.GET['type_name']
        new_type_name = request.GET['new_type_name']
        cloth_list = request.GET['cloth_list']
        print(type_name)
        print(new_type_name)
        print(cloth_list)
    return HttpResponse(json.dumps(resp), content_type="application/json")

    
"""创建设计单
    接口说明:
        创建设计单
    请求方式:
        GET
    传入参数:
        new_type_name: 新设计单名称
        cloth_list: 布料id列表
    响应内容:
        msg: 提示的信息,
        data: {
            is_success: 是否创建成功(True or False),
            text: 创建失败的提示信息
        }
        error: 错误提示信息
"""
def create_sjd(request):
    resp = {
        'msg': '',
        'error': '',
        'data': {
            'is_success': True,
            'text': '创建设计单失败'
        }
    }
    if request.method == 'GET':
        new_type_name = request.GET['new_type_name']
        cloth_list = request.GET['cloth_list']
        print(new_type_name)
        print(cloth_list)
    return HttpResponse(json.dumps(resp), content_type="application/json")

"""删除设计单
    接口说明:
        删除设计单
    请求方式:
        GET
    传入参数:
        type_name: 设计单名称
    响应内容:
        msg: 提示的信息,
        data: {
            is_success: 是否删除成功(True or False),
            text: 删除失败的提示信息
        }
        error: 错误提示信息
"""
def del_sjd(request):
    resp = {
        'msg': '',
        'error': '',
        'data': {
            'is_success': True,
            'text': '删除设计单失败'
        }
    }
    if request.method == 'GET':
        type_name = request.GET['type_name']
        print(type_name)
    return HttpResponse(json.dumps(resp), content_type="application/json")
