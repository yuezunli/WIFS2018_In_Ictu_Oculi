"""
In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
IEEE International Workshop on Information Forensics and Security (WIFS), 2018
Yuezun Li, Ming-ching Chang and Siwei Lyu
"""
from deep_base import ops as net_ops
from deep_base import vgg16 as base
import tensorflow as tf
import numpy as np
import yaml, os
from easydict import EasyDict as edict
pwd = os.path.dirname(__file__)

class BlinkCNN(object):
    """
    CNN for eye blinking detection
    """

    def __init__(self,
                 is_train
                 ):

        cfg_file = os.path.join(pwd, 'blink_cnn.yml')
        with open(cfg_file, 'r') as f:
            cfg = edict(yaml.load(f))

        self.cfg = cfg
        self.img_size = cfg.IMG_SIZE
        self.num_classes = cfg.NUM_CLASS
        self.is_train = is_train

        self.layers = {}
        self.params = {}

    def build(self):
        # Input
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size[0], self.img_size[1], self.img_size[2]])
        self.layers = base.get_prob(self.input, self.params, self.num_classes, self.is_train)
        self.prob = self.layers.prob
        self.gt = tf.placeholder(dtype=tf.int32, shape=[None])
        self.var_list = tf.trainable_variables()

    def loss(self):
        self.net_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.gt, logits=self.layers.fc8)
        self.net_loss = tf.reduce_mean(self.net_loss)
        tf.losses.add_loss(self.net_loss)
        # L2 weight regularize
        self.L2_loss = tf.reduce_mean([self.cfg.TRAIN.BETA * tf.nn.l2_loss(v)
                     for v in tf.trainable_variables() if 'weights' in v.name])
        tf.losses.add_loss(self.L2_loss)
        self.total_loss = tf.losses.get_total_loss()


class BlinkLRCN(object):
    """
    LRCN for eye blinking detection
    """

    def __init__(self,
                 is_train
                 ):

        cfg_file = os.path.join(pwd, 'blink_lrcn.yml')
        with open(cfg_file, 'r') as f:
            cfg = edict(yaml.load(f))

        self.cfg = cfg
        self.img_size = cfg.IMG_SIZE
        self.num_classes = cfg.NUM_CLASS
        self.is_train = is_train

        self.rnn_type = cfg.RNN_TYPE
        self.max_time = cfg.MAX_TIME
        self.hidden_unit = cfg.HIDDEN_UNIT

        if self.is_train:
            self.batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self.batch_size = cfg.TEST.BATCH_SIZE
        self.layers = {}
        self.params = {}

    def build(self):
        self.input = tf.placeholder(dtype=tf.float32,
                                    shape=[self.batch_size, self.max_time, self.img_size[0], self.img_size[1], self.img_size[2]])
        self.blined_gt = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.eye_state_gt = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_time])
        self.seq_len = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

        self.vgg16_fc6 = self._vgg16(self.input)
        self.rnn_out = self._rnn_cell(self.vgg16_fc6)
        self.out = self._fc(self.rnn_out)
        self.prob = tf.nn.softmax(self.out, dim=-1)

    def _vgg16(self, input):
        # Reshape from NxTxHxWxC to (NxT)xHxWxC
        input = tf.reshape(input, [-1, self.img_size[0], self.img_size[1], self.img_size[2]])
        layers = base.get_vgg16_pool5(input, self.params)
        layers.fc6 = net_ops.fully_connected(input=layers.pool5, num_neuron=4096, name='fc6', params=self.params)
        if self.is_train:
            layers.fc6 = tf.nn.dropout(layers.fc6, keep_prob=0.5)
        layers.fc6_relu = net_ops.activate(input=layers.fc6, act_type='relu', name='fc6_relu')
        cnn_out = tf.reshape(layers.fc6_relu, [-1, self.max_time, 4096])
        return cnn_out

    def _rnn_cell(self, input):
        with tf.variable_scope('rnn_cell'):
            size = np.prod(input.get_shape().as_list()[2:])
            rnn_inputs = tf.reshape(input, (-1, self.max_time, size))
            if self.rnn_type == 'LSTM':
                cell = tf.contrib.rnn.LSTMCell(self.hidden_unit)
            elif self.rnn_type == 'GRU':
                cell = tf.contrib.rnn.GRUCell(self.hidden_unit)
            else:
                raise ValueError('We only support LSTM or GRU...')
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell,
                rnn_inputs,
                sequence_length=self.seq_len,
                dtype = tf.float32
            )
            return rnn_outputs

    def _avg_rnn_out(self, rnn_out):
        seq_len = tf.cast(self.seq_len, dtype=tf.float32)
        avg = tf.reduce_sum(rnn_out, axis=1) / tf.expand_dims(seq_len, axis=-1)
        return avg

    def _fc(self, input):
        # Reshape from NxTx256 to (NxT)x256
        input = tf.reshape(input, [-1, self.hidden_unit])
        out = net_ops.fully_connected(input=input, num_neuron=self.num_classes, name='fc_after_rnn', params=self.params)
        out = tf.reshape(out, [-1, self.max_time, self.num_classes])
        return out

    def loss(self):
        self.net_loss = []
        for batch_id in range(self.batch_size):
            out_cur = self.out[batch_id, :, :]
            eye_state_cur = self.eye_state_gt[batch_id, :]
            weights = tf.gather(tf.constant(self.cfg.TRAIN.CLASS_WEIGHTS, dtype=tf.float32), eye_state_cur)
            loss_per_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eye_state_cur, logits=out_cur)  # T x num_class
            loss_per_batch = loss_per_batch * weights
            # Select loss by real len
            seq_len = tf.cast(self.seq_len[batch_id], dtype=tf.float32)
            tf_idx = tf.range(0, self.seq_len[batch_id])
            loss_per_batch = tf.reduce_sum(tf.gather(loss_per_batch, tf_idx, axis=0)) / seq_len
            self.net_loss.append(loss_per_batch)
        self.net_loss = tf.reduce_mean(self.net_loss)
        tf.losses.add_loss(self.net_loss)
        # L2 weight regularize
        self.L2_loss = tf.reduce_mean([self.cfg.TRAIN.BETA * tf.nn.l2_loss(v)
                                       for v in tf.trainable_variables() if 'weights' in v.name or 'kernel' in v.name])
        tf.losses.add_loss(self.L2_loss)
        self.total_loss = tf.losses.get_total_loss()