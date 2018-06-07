#coding=utf-8
import tensorflow as tf
import os
import argparse
from tensorflow.python.framework import graph_util
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import tensorflow as tf
import sys
from tensorflow.contrib import slim
sys.path.insert(0, '/home/lol/DeepLearn/models/research/slim/')
from nets import nets_factory
from datasets import dataset_classification
from preprocessing import preprocessing_factory
from tensorflow.contrib.slim import evaluation
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import monitored_session
from tensorflow.python.summary import summary
import numpy as np
import math

tf.logging.set_verbosity(tf.logging.INFO)

checkpoint = tf.train.get_checkpoint_state('/home/lol/DeepLearn/Tensorflow-Ipynb/logs/')

input_checkpoint = checkpoint.model_checkpoint_path

print input_checkpoint

import cv2
import imutils

with tf.Graph().as_default():

    with tf.Session() as sess:
        tf_global_step = slim.get_or_create_global_step()

        dataset = dataset_classification.get_dataset(dataset_dir='/media/lol/训练数据/TF测试/',
                                                    num_samples=50, num_classes=100000)

        network_fn = nets_factory.get_network_fn('resnet_v1_50', num_classes=100000, is_training=False)

        provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=False)

        [image, label] = provider.get(['image', 'label'])
        
        temp_op = image

        image_preprocessing_fn = preprocessing_factory.get_preprocessing('resnet_v1_50', is_training=False)

        eval_image_size = network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch([image, label], batch_size=8)

        logits, _ = network_fn(images)

        predictions = tf.argmax(logits, 1)

        variables_to_restore = slim.get_variables_to_restore()
        
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # This ensures that we make a single pass over all of the data.
        num_batches = math.ceil(dataset.num_samples / float(8))

        tf.logging.info('Evaluating %s' % input_checkpoint)

        top_k_op = temp_op

        eval_op = list(names_to_updates.values())
        eval_op.append(top_k_op)

        slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=input_checkpoint,
            logdir='/tmp/tfmodel/',
            num_evals=num_batches,
            eval_op=eval_op,
            variables_to_restore=variables_to_restore)