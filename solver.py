"""
In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
IEEE International Workshop on Information Forensics and Security (WIFS), 2018
Yuezun Li, Ming-ching Chang and Siwei Lyu
"""
import tensorflow as tf
import os, cv2
from deep_base.ops import get_restore_var_list
import yaml, os
from easydict import EasyDict as edict
pwd = os.path.dirname(__file__)


class Solver(object):
    """
    Solver for training and testing
    """
    def __init__(self,
                 sess,
                 net,
                 mode='cnn'):
        cfg_file = os.path.join(pwd, 'blink_{}.yml'.format(mode))
        with open(cfg_file, 'r') as f:
            cfg = edict(yaml.load(f))

        self.sess = sess
        self.net = net
        self.cfg = cfg
        self.mode = mode

    def init(self):
        cfg = self.cfg
        self.img_size = cfg.IMG_SIZE
        pwd = os.path.dirname(os.path.abspath(__file__))
        self.summary_dir = os.path.join(pwd, cfg.SUMMARY_DIR)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.model_dir = os.path.join(pwd, cfg.MODEL_DIR)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_path = os.path.join(self.model_dir, 'model.ckpt')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.saver = tf.train.Saver(max_to_keep=5)
        # initialize the graph
        if self.net.is_train:
            self.num_epoch = cfg.TRAIN.NUM_EPOCH
            self.learning_rate = cfg.TRAIN.LEARNING_RATE
            self.decay_rate = cfg.TRAIN.DECAY_RATE
            self.decay_step = cfg.TRAIN.DECAY_STEP
            self.net.loss()
            self.set_optimizer()
            # Add summary
            self.loss_summary = tf.summary.scalar('loss_summary', self.net.total_loss)
            self.lr_summary = tf.summary.scalar('learning_rate_summary', self.LR)
            self.summary = tf.summary.merge([self.loss_summary, self.lr_summary])
            self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.load()

    def train(self, *args):
        if self.mode == 'cnn':
            return self.train_cnn(images=args[0], labels=args[1])
        elif self.mode == 'lrcn':
            return self.train_lrcn(seq_tensor=args[0],
                                   len_list=args[1],
                                   state_list=args[2])
        else:
            raise ValueError('We only support mode = [cnn, lrcn]...')

    def test(self, *args):
        if self.mode == 'cnn':
            return self.test_cnn(images=args[0])
        elif self.mode == 'lrcn':
            return self.test_lrcn(seq_tensor=args[0],
                                   len_list=args[1])
        else:
            raise ValueError('We only support mode = [cnn, lrcn]...')

    def test_cnn(self, images):
        # Check input size
        for i, im in enumerate(images):
            images[i] = cv2.resize(im, (self.img_size[0], self.img_size[1]))

        feed_dict = {
            self.net.input: images,
        }
        fetch_list = [
            self.net.prob,
        ]
        return self.sess.run(fetch_list, feed_dict=feed_dict)

    def train_cnn(self, images, labels):
        feed_dict = {
            self.net.input: images,
            self.net.gt: labels
        }
        fetch_list = [
            self.train_op,
            self.summary,
            self.net.prob,
            self.net.net_loss,
        ]
        return self.sess.run(fetch_list, feed_dict=feed_dict)

    def test_lrcn(self, seq_tensor, len_list):
        feed_dict = {
            self.net.input: seq_tensor,
            self.net.seq_len: len_list,
        }
        fetch_list = [
            self.net.prob,
        ]
        return self.sess.run(fetch_list, feed_dict=feed_dict)

    def train_lrcn(self, seq_tensor, len_list, state_list):
        feed_dict = {
            self.net.input: seq_tensor,
            self.net.seq_len: len_list,
            self.net.eye_state_gt: state_list
        }
        fetch_list = [
            self.train_op,
            self.summary,
            self.net.prob,
            self.net.net_loss,

        ]
        return self.sess.run(fetch_list, feed_dict=feed_dict)

    def save(self, step):
        """ Save checkpoints """
        save_path = self.saver.save(self.sess, self.model_path, global_step=step)
        print('Model {} saved in file.'.format(save_path))

    def load(self):
        """Load weights from checkpoint"""
        if os.path.isfile(self.model_path + '.meta'):
            variables_to_restore = get_restore_var_list(self.model_path)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(self.sess, self.model_path)
            print('Loading checkpoint {}'.format(self.model_path))
        else:
            print('Loading failed.')

    def set_optimizer(self):
        # Set learning rate decay
        self.LR = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_step,
            decay_rate=self.decay_rate,
            staircase=True
        )
        if self.cfg.TRAIN.METHOD == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.LR,
            )
        elif self.cfg.TRAIN.METHOD == 'Adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.LR,
            )
        else:
            raise ValueError('We only support [SGD, Adam] right now...')

        self.train_op = optimizer.minimize(
            loss=self.net.total_loss,
            global_step=self.global_step,
            var_list=None)


