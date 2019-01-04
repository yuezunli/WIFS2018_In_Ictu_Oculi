import numpy as np
import cv2


def vis_eye(im, left_eye, right_eye):
    leftEyeHull = cv2.convexHull(left_eye)
    rightEyeHull = cv2.convexHull(right_eye)
    cv2.drawContours(im, [leftEyeHull], -1, (255, 0, 0), 1)
    cv2.drawContours(im, [rightEyeHull], -1, (255, 0, 0), 1)
    center_left_ear = np.mean(left_eye, axis=0).astype(int)
    center_right_ear = np.mean(right_eye, axis=0).astype(int)
    cv2.circle(im, (center_left_ear[0], center_left_ear[1]), 2, (0, 255, 0), -1)
    cv2.circle(im, (center_right_ear[0], center_right_ear[1]), 2, (0, 255, 0), -1)
    return im

def vis_seq(batch, len_list, dir):
    # concatenate together
    for b in range(len(batch)):
        seq_len = len_list[b]
        seq = batch[b][:seq_len]
        im = np.concatenate(seq, axis=1)
        cv2.imwrite(dir + '/' + str(b) + '.jpg', im)

def vis_im(batch, dir):
    # concatenate together
    for b in range(len(batch)):
        im = batch[b]
        cv2.imwrite(dir + '/' + str(b) + '.jpg', im)