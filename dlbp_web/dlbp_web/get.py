# coding=utf-8
from django.http import HttpResponse
from django.shortcuts import render
import json
import cv2
import numpy as np

from multiprocessing.connection import Client
from multiprocessing.connection import Listener

my_id = 99

def uploadImg(request):
    content = {
        'imgs': [],
        'sk_id': 0
    }
    if request.method == 'POST':
        f=request.FILES['img']

        sk_id = request.POST['sk_id']

        import time

        ti = time.time()

        file_name = '/home/lee/DeepLearn/My_BP_TrainCode/dlbp_web/static/image/temp_{}.jpg'.format(ti)
        with open(file_name, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)

        c = Client('/usr/local/server%d.temp' % 99, authkey=b'lee123456')
        # 将信息传送给服务端
        c.send(['-f', file_name, '-k', 50])
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

        result_list = ar[t_idx]

        for i in result_list:
            file_name = i[1]
            i[1] = '/medias/' + file_name.split('/')[-2] + '/' + file_name.split('/')[-1].split('_')[0] + '_000' + file_name.split('/')[-1].split('_')[1][3:]
            i[2] = '%.2f%%' % i[2]

        content = {
            'test_img': '/medias/temp_{}.jpg'.format(ti),
            'imgs': result_list,
            'sk_id': sk_id
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
