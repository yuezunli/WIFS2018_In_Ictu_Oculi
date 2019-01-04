"""
In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
IEEE International Workshop on Information Forensics and Security (WIFS), 2018
Yuezun Li, Ming-ching Chang and Siwei Lyu
"""
import tensorflow as tf
import argparse
import sys
sys.path.append('..')
from solu_base import Solu
from blink_net import BlinkCNN
from solver import Solver


def main(input_vid_path, out_dir):
    solution = Solu(input_vid_path)

    net = BlinkCNN(
        is_train=False
    )
    net.build()
    sess = tf.Session()
    # Init solver
    solver = Solver(sess=sess,
                    net=net)
    solver.init()

    for i in range(solution.frame_num):
        print('Frame: ' + str(i))
        eye1, eye2 = solution.get_eye_by_fid(i)
        if eye1 is not None:
            eye1_prob, = solver.test([eye1])
        else:
            eye1_prob = 0.5

        if eye2 is not None:
            eye2_prob, = solver.test([eye2])
        else:
            eye2_prob = 0.5

        solution.push_eye_prob(eye1_prob[0, 1], eye2_prob[0, 1])
        solution.plot_by_fid(i)

    sess.close()
    tf.reset_default_graph()
    solution.gen_videos(out_dir, 'cnn')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_vid_path', type=str)
    parser.add_argument('--out_dir', type=str)

    args = parser.parse_args()

    main(args.input_vid_path, args.out_dir)
