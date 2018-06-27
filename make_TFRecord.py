#coding=utf-8

from tools import file_tools
import sys
import time

def make_TFRecord(list_file_path, tfrecord_file_path=None):
    from datasets import dataset_utils
    from PIL import Image
    import tensorflow as tf
    import time
    
    # 默认情况下保存为列表名_tfrecord.tfrecords
    if tfrecord_file_path is None:
        tfrecord_file_path = file_tools.getFileName(list_file_path).split('.')[0] + '_tfrecord.tfrecord'

    # 图片列表文件
    list_file = open(list_file_path)

    # 获取每行，并且分隔
    lines = [line.split() for line in list_file]

    list_file.close()

    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_file_path)

    def change_size(image):
        smallest_side = tf.convert_to_tensor(512, dtype=tf.int32)

        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height = tf.to_float(height)
        width = tf.to_float(width)
        smallest_side = tf.to_float(smallest_side)

        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
        new_height = tf.to_int32(tf.rint(height * scale))
        new_width = tf.to_int32(tf.rint(width * scale))
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                                align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        return new_height, new_width, resized_image

    last_time = time.time()
    with tf.Graph().as_default():
        # 读取格式信息
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        with tf.Session('') as sess:
            num = 0
            max_num = len(lines)
            for line in lines:
                # 处理路径名，去除空格的影响
                if len(line) != 2:
                    for index, i in enumerate(line):
                        if index != 0 and index < len(line) - 1:
                            line[0] = line[0] + ' ' + i
                    line[1] = line[len(line) - 1]
                # 判断是否为数字
                if line[1].isdigit():
                    image_data = tf.gfile.FastGFile(line[0], 'rb').read()
                    image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
                    
                    height, width, image = change_size(decode_jpeg)
                    image = sess.run(image, feed_dict={decode_jpeg_data: image_data})
                    example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, int(line[1]))
                    tfrecord_writer.write(example.SerializeToString())
                    num = num + 1
                    if num % 1000 == 0:
                        spend_time = float(1000) / (time.time() - last_time)
                        print('Finish %d/%d(%.2f/sec)........%.2f' % (num, max_num, spend_time, max_num / spend_time))
                        last_time = time.time()
                        # 防止每个tfrecord太大，进行分割
                        tfrecord_writer.close()
                        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_file_path.replace('.', '_%d.' % int(num / 1000)))
            tfrecord_writer.close()


def help():
    print '用法: -f [图片路径] [选项]... [选项]...'
    print ''
    print '将列表中的图片和Label存到TFRecord中(每1000个图片为一个TFrecord)'
    print ''
    print '必选参数.'
    print '-f 指定列表文件，由 make_label.py 生成'
    print '-m 指定tensorflow/models路径，如果没有，请使用  git clone https://github.com/tensorflow/models'
    print ''
    print '可选参数.'
    print '-t 指定生成的TFRecord的路径，默认为list_name_tfrecord.tfrecord'
    print ''

if __name__ == '__main__':
    import sys
    import os
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], 'f:t:m:h')

    list_file_path = None
    tfrecord_file_path = None
    models_path = None

    for op, val in opts:
        if op == '-f':
            list_file_path = val
        elif op == '-t':
            tfrecord_file_path = val
        elif op == '-m':
            models_path = val
        elif op == '-h':
            help()
    
    if list_file_path is None:
        help()
        print '必须使用 -f 指定列表文件'
    elif models_path is None:
        help()
        print '必须使用 -m 指定tensorflow/models路径，如果没有，请使用  git clone https://github.com/tensorflow/models'
    else:
        sys.path.insert(0, models_path + '/research/slim/') #把后面的路径插入到系统路径中 idx=0
        print('%s is Loaded' % (models_path + '/research/slim/'))
        make_TFRecord(list_file_path, tfrecord_file_path)




