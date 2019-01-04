# ================================
# Wrapper for common utils in DNN
# ================================

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.ops import init_ops


# =========================================
# =========================================
# Common operation in network
# =========================================
# =========================================
def activate(input, name, act_type='relu'):
    with tf.variable_scope(name) as scope:
        if act_type == 'relu':
            out = tf.nn.relu(input)
        elif act_type == 'sigmod':
            out = tf.nn.sigmod(input)
        else:
            raise ValueError('act_type is not valid.')

    return out

def conv2D(input,
           shape,
           name,
           padding='SAME',
           strides=(1, 1),
           weights_initializer=xavier_initializer(),
           bias_initializer=init_ops.zeros_initializer(),
           weights_regularizer=None,
           bias_regularizer=None,
           params={}
           ):
    use_bias = use_bias_helper(bias_initializer)
    with tf.variable_scope(name) as scope:
        channel = input.get_shape().as_list()[-1]
        kernel = tf.get_variable(
            name='weights',
            shape=[shape[0], shape[1], channel, shape[2]],
            dtype=tf.float32,
            initializer=weights_initializer,
            regularizer=weights_regularizer
        )
        strides = [1, strides[0], strides[1], 1]
        out = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

        bias = None
        if use_bias:
            bias = tf.get_variable(
                name='biases',
                shape=[shape[2]],
                dtype=tf.float32,
                initializer=bias_initializer,
                regularizer=bias_regularizer
            )
            out = tf.nn.bias_add(out, bias)

        print('{} weights: {}, bias: {}, out: {}'.format(name, kernel, bias, out))
        params[name] = [kernel, bias]

    return out

# =========================================
def max_pool(input,
            name,
            ksize=(2, 2),
            strides=(2, 2),
            padding='SAME'
            ):
    with tf.variable_scope(name) as scope:
        ksize = [1, ksize[0], ksize[1], 1]
        strides = [1, strides[0], strides[1], 1]
        out = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)
        print('{} max pool out: {}'.format(name, out))

    return out

# =========================================
def avg_pool(input,
             name,
             ksize=(2, 2),
             strides=(2, 2),
             padding='SAME'
             ):
    with tf.variable_scope(name) as scope:
        ksize = [1, ksize[0], ksize[1], 1]
        strides = [1, strides[0], strides[1], 1]
        out = tf.nn.avg_pool(input, ksize=ksize, strides=strides, padding=padding)
        print('{} avg pool out: {}'.format(name, out))
    return out


# =========================================
def fully_connected(input,
                    num_neuron,
                    name,
                    weights_initializer=xavier_initializer(),
                    bias_initializer=init_ops.zeros_initializer(),
                    weights_regularizer=None,
                    bias_regularizer=None,
                    params={}
                    ):
    use_bias = use_bias_helper(bias_initializer)
    with tf.variable_scope(name) as scope:
        input_dim = int(np.prod(input.get_shape().as_list()[1:]))
        kernel = tf.get_variable(
            name='weights',
            shape=[input_dim, num_neuron],
            dtype=tf.float32,
            initializer=weights_initializer,
            regularizer=weights_regularizer
        )
        flat = tf.reshape(input, [-1, input_dim])
        out = tf.matmul(flat, kernel)
        bias = None
        if use_bias:
            bias = tf.get_variable(
                name='biases',
                shape=num_neuron,
                dtype=tf.float32,
                initializer=bias_initializer,
                regularizer=bias_regularizer
            )
            out = tf.nn.bias_add(out, bias)

        print('{} weights: {}, bias: {}, out: {}'.format(name, kernel, bias, out))
        params[name] = [kernel, bias]

    return out


# =========================================
def batch_norm(input, name, is_train=True, params={}):
    batch_norm_out = tf.contrib.layers.batch_norm(inputs=input, scale=True, is_training=is_train, scope=name)
    # Get gamma and beta
    # trainable_vars = tf.trainable_variables()
    # gamma = [var for var in trainable_vars if name in var.name and 'gamma' in var.name]
    # beta = [var for var in trainable_vars if name in var.name and 'beta' in var.name]
    # for i in tf.get_default_graph().get_operations():
    #     print i.name
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)  # beta, gamma, moving_mean, moving_variance
    params[name] = var_list
    print('{} {}'.format(name, batch_norm_out))
    return batch_norm_out

# =========================================
def use_bias_helper(bias_initializer):
    """
    Determine if a layer needs bias
    :param bias_initializer:
    :return:
    """
    if bias_initializer is None:
        return False
    else:
        return True

# ============================================
def get_restore_var_list(path):
    """
    Get variable list when restore from ckpt. This is mainly for transferring model to another network
    """
    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # Variables in graph
    saved_vars = list_vars_in_ckpt(path)
    saved_vars_name = [var[0] for var in saved_vars]
    restore_var_list = [var for var in global_vars if var.name[:-2] in saved_vars_name]# or 'vgg_' + var.name[:-2] in saved_vars_name]

    return restore_var_list


# ============================================
def list_vars_in_ckpt(path):
    """List all variables in checkpoint"""
    saved_vars = tf.contrib.framework.list_variables(path)  # List of tuples (name, shape)
    return saved_vars
