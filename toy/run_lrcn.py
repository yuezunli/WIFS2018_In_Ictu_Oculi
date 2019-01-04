"""
In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
IEEE International Workshop on Information Forensics and Security (WIFS), 2018
Yuezun Li, Ming-ching Chang and Siwei Lyu
"""
import tensorflow as tf
import argparse
import numpy as np
import sys
sys.path.append('..')
from solu_base import Solu
from blink_net import BlinkLRCN
from solver import Solver
import cv2
from py_utils import x_utils as ulib


def main(input_vid_path, out_dir):
    solution = Solu(input_vid_path)

    net = BlinkLRCN(
        is_train=False
    )
    net.build()
    sess = tf.Session()
    # Init solver
    solver = Solver(sess=sess,
                    net=net,
                    mode='lrcn')
    solver.init()

    stride = 10
    batch_size = np.arange(0, solution.frame_num, stride)

    for i in batch_size:

        eye1_list, eye2_list = [], []
        eye1_index = []
        eye2_index = []

        for j in range(i, np.minimum(i + stride, solution.frame_num)):
            eye1, eye2 = solution.get_eye_by_fid(j)
            if eye1 is not None:
                eye1_index.append(j)
                eye1_list.append(cv2.resize(eye1, (net.img_size[0], net.img_size[1])))

            if eye2 is not None:
                eye2_index.append(j)
                eye2_list.append(cv2.resize(eye2, (net.img_size[0], net.img_size[1])))

        eye1_full = ulib.pad_to_max_len(eye1_list, net.max_time,
                                        pad=np.zeros(eye1_list[0].shape, dtype=np.int32))
        eye2_full = ulib.pad_to_max_len(eye2_list, net.max_time,
                                        pad=np.zeros(eye2_list[0].shape, dtype=np.int32))
        eye1_probs, = solver.test([eye1_full], [len(eye1_list)])
        eye2_probs, = solver.test([eye2_full], [len(eye2_list)])

        for j in range(i, np.minimum(i + stride, solution.frame_num)):
            if j in eye1_index:
                eye1_prob = eye1_probs[0][eye1_index.index(j), 1]
            else:
                eye1_prob = 0.5

            if j in eye2_index:
                eye2_prob = eye2_probs[0][eye2_index.index(j), 1]
            else:
                eye2_prob = 0.5

            solution.push_eye_prob(eye1_prob, eye2_prob)
            solution.plot_by_fid(j)

    sess.close()
    solution.gen_videos(out_dir, 'lrcn')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_vid_path', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    main(args.input_vid_path, args.out_dir)
