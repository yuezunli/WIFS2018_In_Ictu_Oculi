# =============================
# VGG16 network structure
# =============================

import tensorflow as tf
from easydict import EasyDict as edict

import ops


def get_vgg16_conv5(input, params):
    layers = edict()

    layers.conv1_1 = ops.conv2D(input=input, shape=(3, 3, 64), name='conv1_1', params=params)
    layers.conv1_1_relu = ops.activate(input=layers.conv1_1, name='conv1_1_relu', act_type='relu')
    layers.conv1_2 = ops.conv2D(input=layers.conv1_1_relu, shape=(3, 3, 64), name='conv1_2', params=params)
    layers.conv1_2_relu = ops.activate(input=layers.conv1_2, name='conv1_2_relu', act_type='relu')
    layers.pool1 = ops.max_pool(input=layers.conv1_2_relu, name='pool1')

    layers.conv2_1 = ops.conv2D(input=layers.pool1, shape=(3, 3, 128), name='conv2_1', params=params)
    layers.conv2_1_relu = ops.activate(input=layers.conv2_1, name='conv2_1_relu', act_type='relu')
    layers.conv2_2 = ops.conv2D(input=layers.conv2_1_relu, shape=(3, 3, 128), name='conv2_2', params=params)
    layers.conv2_2_relu = ops.activate(input=layers.conv2_2, name='conv2_2_relu', act_type='relu')
    layers.pool2 = ops.max_pool(input=layers.conv2_2_relu, name='pool2')

    layers.conv3_1 = ops.conv2D(input=layers.pool2, shape=(3, 3, 256), name='conv3_1', params=params)
    layers.conv3_1_relu = ops.activate(input=layers.conv3_1, name='conv3_1_relu', act_type='relu')
    layers.conv3_2 = ops.conv2D(input=layers.conv3_1_relu, shape=(3, 3, 256), name='conv3_2', params=params)
    layers.conv3_2_relu = ops.activate(input=layers.conv3_2, name='conv3_2_relu', act_type='relu')
    layers.conv3_3 = ops.conv2D(input=layers.conv3_2_relu, shape=(3, 3, 256), name='conv3_3', params=params)
    layers.conv3_3_relu = ops.activate(input=layers.conv3_3, name='conv3_3_relu', act_type='relu')
    layers.pool3 = ops.max_pool(input=layers.conv3_3_relu, name='pool3')

    layers.conv4_1 = ops.conv2D(input=layers.pool3, shape=(3, 3, 512), name='conv4_1', params=params)
    layers.conv4_1_relu = ops.activate(input=layers.conv4_1, name='conv4_1_relu', act_type='relu')
    layers.conv4_2 = ops.conv2D(input=layers.conv4_1_relu, shape=(3, 3, 512), name='conv4_2', params=params)
    layers.conv4_2_relu = ops.activate(input=layers.conv4_2, name='conv4_2_relu', act_type='relu')
    layers.conv4_3 = ops.conv2D(input=layers.conv4_2_relu, shape=(3, 3, 512), name='conv4_3', params=params)
    layers.conv4_3_relu = ops.activate(input=layers.conv4_3, name='conv4_3_relu', act_type='relu')
    layers.pool4 = ops.max_pool(input=layers.conv4_3_relu, name='pool4')

    layers.conv5_1 = ops.conv2D(input=layers.pool4, shape=(3, 3, 512), name='conv5_1', params=params)
    layers.conv5_1_relu = ops.activate(input=layers.conv5_1, name='conv5_1_relu', act_type='relu')
    layers.conv5_2 = ops.conv2D(input=layers.conv5_1_relu, shape=(3, 3, 512), name='conv5_2', params=params)
    layers.conv5_2_relu = ops.activate(input=layers.conv5_2, name='conv5_2_relu', act_type='relu')
    layers.conv5_3 = ops.conv2D(input=layers.conv5_2_relu, shape=(3, 3, 512), name='conv5_3', params=params)
    layers.conv5_3_relu = ops.activate(input=layers.conv5_3, name='conv5_3_relu', act_type='relu')

    return layers

def get_vgg16_pool5(input, params):
    layers = get_vgg16_conv5(input, params)
    layers.pool5 = ops.max_pool(input=layers.conv5_3_relu, name='pool5')

    return layers

def get_prob(input, params, num_class=1000, is_train=True):
    # Get pool5
    layers = get_vgg16_pool5(input, params)
    layers.fc6 = ops.fully_connected(input=layers.pool5, num_neuron=4096, name='fc6', params=params)
    if is_train:
        layers.fc6 = tf.nn.dropout(layers.fc6, keep_prob=0.5)
    layers.fc6_relu = ops.activate(input=layers.fc6, act_type='relu', name='fc6_relu')
    layers.fc7 = ops.fully_connected(input=layers.fc6_relu, num_neuron=4096, name='fc7', params=params)
    if is_train:
        layers.fc7 = tf.nn.dropout(layers.fc7, keep_prob=0.5)
    layers.fc7_relu = ops.activate(input=layers.fc7, act_type='relu', name='fc7_relu')
    layers.fc8 = ops.fully_connected(input=layers.fc7_relu, num_neuron=num_class, name='fc8', params=params)
    layers.prob = tf.nn.softmax(layers.fc8)
    return layers