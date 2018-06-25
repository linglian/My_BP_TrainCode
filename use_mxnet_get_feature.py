#coding=utf-8
import OpenSSL.SSL
import mxnet as mx
import cv2
import os
import numpy as np

def use_mxnet_get_feature(base_path, prefix='resnet-50', layer='flatten0_output'):
    model = mx.model.FeedForward.load(prefix=prefix, epoch=0, ctx=mx.gpu(), numpy_batch_size=1)
    internals = model.symbol.get_internals()
    feature_symbol = internals[layer]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(0), symbol=feature_symbol, numpy_batch_size=1,
                                            arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)

    def getFeature(image_filepath, feature_extractor):
        import cv2
        import numpy as np
        def getImage(img):
            import numpy as np
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            img = img[np.newaxis, :]
            return img
        img = cv2.imread(image_filepath)
        if img.shape[0] != 224 or img.shape[1] != 224:
            print 'Must be resize'
            img = cv2.resize(img, (224, 224))
        img = getImage(img)
        result_feature = feature_extractor.predict(img)
        return np.ravel(result_feature)

    def format_feature(class_name, feature, image_filepath):
        return [feature, class_name, image_filepath]

    class_name_and_path_list = [[floder, os.path.join(base_path, floder)] for floder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, floder))]
    
    max_num = 0
    for class_name_and_path in class_name_and_path_list:
        image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith('.jpg')]
        max_num = max_num + len(image_path_list)
        
    now_num = 0
    for class_name_and_path in class_name_and_path_list:
        if os.path.exists(os.path.join(class_name_and_path[1], 'resnet-50_feature.npy')):
            continue
        image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith('.jpg')]
        feature_list = []
        for image_path in image_path_list:
            feature_list.append(format_feature(feature=getFeature(image_filepath=image_path,
                                                                feature_extractor=feature_extractor),
                                            class_name=class_name_and_path[0], image_filepath=image_path))
            now_num = now_num + 1
        print '正在提取中...%d / %d' % (now_num, max_num)
        np.save(os.path.join(class_name_and_path[1], 'resnet-50_feature.npy'), feature_list)

    max_num = 0
    for class_name_and_path in class_name_and_path_list:
        image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith('.jpg')]
        max_num = max_num + len(image_path_list)
        
    now_num = 0

    feature_list = []
    class_name_list = []
    file_path_list = []

    for class_name_and_path in class_name_and_path_list:
        image_path_list = [os.path.join(class_name_and_path[1], image_file) for image_file in os.listdir(class_name_and_path[1]) if image_file.endswith('.jpg')]
        npy = np.load(os.path.join(class_name_and_path[1], 'resnet-50_feature.npy'))
        for t in npy:
            feature_list.append(t[0])
            class_name_list.append(t[1])
            file_path_list.append(t[2])
            now_num = now_num + 1
        print '正在提取中...%d / %d' % (now_num, max_num)

    np.save(os.path.join(base_path, 'feature.npy'), feature_list)
    np.save(os.path.join(base_path, 'class_name.npy'), class_name_list)
    np.save(os.path.join(base_path, 'file_path.npy'), file_path_list)

def help():
    print '用法: -f [图片路径] [选项]... [选项]...'
    print ''
    print '将文件夹内的指定格式的图片文件进行特征提取，并保存成npy格式的文件，保存格式为[feature, class_name, image_filepath]'
    print ''
    print '必选参数.'
    print '-f 指定图片所在文件夹路径'
    print ''
    print '可选参数.'
    print '-p 指定模型，默认为resnet-50'
    print '-l 指定层，默认为flatten0_output'
    print ''

if __name__ == '__main__':
    
    import sys
    import getopt

    base_path = None
    prefix='resnet-50'
    layer='flatten0_output'

    opts, args = getopt.getopt(sys.argv[1:], 'f:p:l:h')

    for op, value in opts:
        if op == '-f':
            base_path = value
        elif op == '-p':
            prefix = value
        elif op == '-l':
            layer = value
        elif op == '-h':
            help()
    
    if base_path is None:
        help()
        print '必须使用 -f 指定图片路径'
    else:
        use_mxnet_get_feature(base_path, prefix, layer)

