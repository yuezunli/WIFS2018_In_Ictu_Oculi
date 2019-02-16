"""
In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
IEEE International Workshop on Information Forensics and Security (WIFS), 2018
Yuezun Li, Ming-ching Chang and Siwei Lyu
"""

from blink_net import BlinkLRCN
from solver import Solver
import tensorflow as tf
import numpy as np
from proc_data.seq_data import SeqData
import py_utils.vis_utils.vis as uvis
from pprint import pprint


def main():

    with tf.Session() as sess:
        # Build network
        net = BlinkLRCN(is_train=True)
        net.build()

        # Init solver
        solver = Solver(sess=sess, net=net, mode='lrcn')
        solver.init()

        # Eye state data generator
        data_gen = SeqData(
            anno_path='sample_sq_data/train.p',
            data_dir='sample_sq_data/',
            batch_size=net.cfg.TRAIN.BATCH_SIZE,
            max_seq_len=net.cfg.MAX_TIME,
            is_augment=True,
            is_shuffle=True
        )

        print('Training...')
        # Training
        batch_num = data_gen.batch_num
        summary_idx = 0
        for epoch in range(solver.cfg.TRAIN.NUM_EPOCH):
            for i in range(batch_num):
                seq_tensor, len_list, scores_list, state_list, label_list, seq_name_list \
                    = data_gen.get_batch(i, size=net.cfg.IMG_SIZE[:2])
                uvis.vis_seq(seq_tensor, len_list, 'tmp')
                _, summary, prob, net_loss = solver.train(seq_tensor, len_list, state_list)
                pred_state_list = np.argmax(prob, axis=-1)
                solver.writer.add_summary(summary, summary_idx)
                summary_idx += 1
                list_1, list_2 = [], []
                for j, L in enumerate(len_list):
                    list_1.append(state_list[j][:L])
                    list_2.append(pred_state_list[j][:L])
                print('====================================')
                print('Net loss: {}'.format(net_loss))
                print('Real state:')
                pprint(list_1)
                print('Pred state:')
                pprint(list_2)
                print('epoch: {}, batch_idx: {}'.format(epoch, i))
            if epoch % solver.cfg.TRAIN.SAVE_INTERVAL == 0:
                solver.save(epoch)

if __name__ == '__main__':
    main()