"""
In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
IEEE International Workshop on Information Forensics and Security (WIFS), 2018
Yuezun Li, Ming-ching Chang and Siwei Lyu
--------------------------
Code written by Yuezun Li
"""

from blink_net import BlinkCNN
from solver import Solver
import tensorflow as tf
import numpy as np
from proc_data.eye_data import EyeData
import py_utils.vis_utils.vis as uvis


def main():

    with tf.Session() as sess:
        # Build network
        net = BlinkCNN(is_train=True)
        net.build()

        # Init solver
        solver = Solver(sess=sess, net=net)
        solver.init()

        # Eye state data generator
        data_gen = EyeData(
            anno_path='sample_eye_data/train.p',
            data_dir='sample_eye_data/',
            batch_size=net.cfg.TRAIN.BATCH_SIZE,
            is_augment=True,
            is_shuffle=True
        )

        print('Training...')
        # Training
        batch_num = data_gen.batch_num
        summary_idx = 0
        for epoch in range(solver.cfg.TRAIN.NUM_EPOCH):
            for i in range(batch_num):
                im_list, label_list, im_name_list \
                    = data_gen.get_batch(i, size=net.cfg.IMG_SIZE[:2])
                uvis.vis_im(im_list, 'tmp')
                _, summary, prob, net_loss = solver.train(im_list, label_list)
                solver.writer.add_summary(summary, summary_idx)
                summary_idx += 1
                pred_label = np.argmax(prob, axis=-1)
                print('====================================')
                print('Net loss: {}'.format(net_loss))
                print('Real label: {}'.format(label_list))
                print('Pred label: {}'.format(pred_label))
                print('Epoch: {}'.format(epoch))
            if epoch % solver.cfg.TRAIN.SAVE_INTERVAL == 0:
                solver.save(epoch)

if __name__ == '__main__':
    main()