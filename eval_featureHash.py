#coding=utf-8
import numpy as np
import os
import falconn

global is_need_init_find_k_FeatureHash
global falconn_query, mxnet_feature_extractor
global my_feature, my_class_name, my_file_path

is_need_init_find_k_FeatureHash = True

def find_k_FeatureHash(model_path, image_file_path, k):
    global is_need_init_find_k_FeatureHash
    global falconn_query, mxnet_feature_extractor
    global my_feature, my_class_name, my_file_path
    
    def getFeature(image_filepath, feature_extractor):
        import cv2
        def getImage(img):
            import numpy as np
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            img = img[np.newaxis, :]
            return img
        img = cv2.imread(image_filepath)
        if img.shape[0] != 224 or img.shape[1] != 224:
            img = cv2.resize(img, (224, 224))
        img = getImage(img)
        result_feature = feature_extractor.predict(img)
        return np.ravel(result_feature)
    
    if is_need_init_find_k_FeatureHash:
        print '正在进行初始化...'
        def init_falconn():
            dim = 2048
            # 获得数组
            my_feature = np.load(os.path.join(model_path, 'feature.npy'))
            my_class_name = np.load(os.path.join(model_path, 'class_name.npy'))
            my_file_path = np.load(os.path.join(model_path, 'file_path.npy'))
            # 获取数组数量
            trainNum=len(my_feature)
            # 获得默认参数
            p=falconn.get_default_parameters(trainNum, dim)
            t=falconn.LSHIndex(p)
            dataset = my_feature
            # 生成hash
            t.setup(dataset)
            q = t.construct_query_pool()
            return my_feature, my_class_name, my_file_path, q
        def init_mxnet():
            import mxnet as mx
            model = mx.model.FeedForward.load(prefix='resnet-50', epoch=0, ctx=mx.gpu(), numpy_batch_size=1)
            internals = model.symbol.get_internals()
            feature_symbol = internals['flatten0_output']
            feature_extractor = mx.model.FeedForward(ctx=mx.gpu(0), symbol=feature_symbol, numpy_batch_size=1,
                                                     arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)
            return feature_extractor
        my_feature, my_class_name, my_file_path, falconn_query = init_falconn()
        mxnet_feature_extractor = init_mxnet()
        is_need_init_find_k_FeatureHash = False
        print '初始化结束...'
    featrue = getFeature(image_file_path, feature_extractor=mxnet_feature_extractor)
    find_list = falconn_query.find_k_nearest_neighbors(query=featrue, k=k)
    class_name_list = my_class_name[find_list]
    file_path_list = my_file_path[find_list]
    return class_name_list, file_path_list

def eval_featureHash(base_path, model_path, k = [1, 5, 10], image_format='.webp'):
    def test_featureHash(bashpath, model_path, k, image_format):

        class_name_and_path_list = [[floder, os.path.join(base_path, floder)] for floder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, floder))]

        my_sum = 0

        for class_name_and_path in class_name_and_path_list:
            image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith(image_format)]
            my_sum = my_sum + len(image_path_list)
        
        print my_sum
        all_bad_list = []
        if isinstance(k, int):
            k = [k]
        for tk in k:
            num = 0
            now_num = 0
            bad_list = []
            for class_name_and_path in class_name_and_path_list:
                image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith(image_format)]
                class_name = class_name_and_path[0]
                for image_path in image_path_list:
                    c, f = find_k_FeatureHash(model_path, image_file_path=image_path, k=tk)
                    is_find = False
                    now_num = now_num + 1
                    for tc in c:
                        if tc == class_name:
                            num = num + 1
                            is_find = True
                            break
                    if is_find is False:
                        bad_list.append(image_path)
                print 'Rank=%d时的正确率为%.02f(%d/%d)' % (tk, float(num) / now_num, num, my_sum)
            all_bad_list.append(bad_list)
        return all_bad_list
    all_bad_list = test_featureHash(base_path, model_path, k, image_format)
    np.save('./all_bad_list.npy', all_bad_list)


def help():
    print '用法: -f [图片路径] [选项]... [选项]...'
    print ''
    print '将文件夹内的指定格式的图片文件进行FeatureHash测试'
    print ''
    print '必选参数.'
    print '-f 指定图片所在文件夹路径'
    print '-m 指定三个npy文件所在路径'
    print ''
    print '可选参数.'
    print '-r 指定Rank 默认为[1, 5, 10]'
    print '-t 指定图片格式'
    print ''

if __name__ == '__main__':
    
    import sys
    import getopt

    model_path = None
    base_path = None
    rank = [1, 5, 10]
    image_format = '.webp'

    opts, args = getopt.getopt(sys.argv[1:], 'f:m:r:t:h')

    for op, value in opts:
        if op == '-f':
            base_path = value
        elif op == '-m':
            model_path = value
        elif op == '-r':
            rank = value
        elif op == '-t':
            image_format = value
        elif op == '-h':
            help()
    
    if model_path is None:
        help()
        print '必须使用 -m 指定三个npy所在路径'
    elif base_path is None:
        help()
        print '必须使用 -f 指定图片路径'
    else:
        eval_featureHash(base_path, model_path, rank, image_format)

