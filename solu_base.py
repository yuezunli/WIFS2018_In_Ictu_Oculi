"""
In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking
IEEE International Workshop on Information Forensics and Security (WIFS), 2018
Yuezun Li, Ming-ching Chang and Siwei Lyu
"""

import numpy as np
import cv2, os
import matplotlib
import sys
import warnings
import dlib
matplotlib.use('Agg')
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
from py_utils.face_utils import lib
from py_utils.vid_utils import proc_vid as pv
from py_utils.plot_utils import plot


class Solu(object):

    def __init__(self,
                 input_vid_path,
                 output_height=300,
                 ):
        # Input video
        self.input_vid_path = input_vid_path
        # parse video
        print('Parsing video {}'.format(str(self.input_vid_path)))
        self.imgs, self.frame_num, self.fps, self.img_w, self.img_h = pv.parse_vid(str(self.input_vid_path))
        if len(self.imgs) != self.frame_num:
            warnings.warn('Frame number is not consistent with the number of images in video...')
            self.frame_num = len(self.imgs)
        print('Eye blinking solution is building...')

        self._set_up_dlib()

        self.output_height = output_height
        factor = float(self.output_height) / self.img_h

        # Resize imgs for final video generation
        # Resize self.imgs according to self.output_height
        self.aligned_imgs = []
        self.left_eyes = []
        self.right_eyes = []

        self.resized_imgs = []
        print('face aligning...')
        for i, im in enumerate(tqdm(self.imgs)):
            face_cache = lib.align(im[:, :, (2,1,0)], self.front_face_detector, self.lmark_predictor)
            if len(face_cache) == 0:
                self.left_eyes.append(None)
                self.right_eyes.append(None)
                continue

            if len(face_cache) > 1:
                raise ValueError('{} faces are in image, we only support one face in image.')

            aligned_img, aligned_shapes_cur = lib.get_aligned_face_and_landmarks(im, face_cache)
            # crop eyes
            leye, reye = lib.crop_eye(aligned_img[0], aligned_shapes_cur[0])
            self.left_eyes.append(leye)
            self.right_eyes.append(reye)
            im_resized = cv2.resize(im, None, None, fx=factor, fy=factor)
            self.resized_imgs.append(im_resized)

        # For visualize
        self.plot_vis_list = []
        self.total_eye1_prob = []
        self.total_eye2_prob = []

    def _set_up_dlib(self):
        # Note that CUDA dir should be /usr/local/cuda and without AVX intructions
        pwd = os.path.dirname(os.path.abspath(__file__))
        # self.cnn_face_detector = dlib.cnn_face_detection_model_v1(pwd + '/mmod_human_face_detector.dat')
        self.front_face_detector = dlib.get_frontal_face_detector()
        self.lmark_predictor = dlib.shape_predictor(pwd + '/dlib_model/shape_predictor_68_face_landmarks.dat')

    def gen_videos(self, out_dir, tag=''):
        vid_name = os.path.basename(self.input_vid_path)
        out_path = os.path.join(out_dir, tag + '_' + vid_name)
        print('Generating video: {}'.format(out_path))
        # Output folder
        if not os.path.exists(os.path.dirname(out_dir)):
            os.makedirs(os.path.dirname(out_dir))

        final_list = []
        for i in tqdm(range(self.frame_num)):
            final_vis = np.concatenate([self.resized_imgs[i], self.plot_vis_list[i]], axis=1)
            final_list.append(final_vis)
        pv.gen_vid(out_path, final_list, self.fps)

    def get_eye_by_fid(self, i):
        eye1, eye2 = self.left_eyes[i], self.right_eyes[i]
        return eye1, eye2

    def push_eye_prob(self, eye1_prob, eye2_prob):
        self.total_eye1_prob.append(eye1_prob)
        self.total_eye2_prob.append(eye2_prob)

    def plot_by_fid(self, i):
        # Vis plots
        max_X = self.frame_num / self.fps
        params = {}
        params['title'] = 'Eye-state-probability'
        params['colors'] = ['b-']
        params['markers'] = [None]
        params['linewidth'] = 3
        params['markersize'] = None
        params['figsize'] = None

        x_axis = np.arange(self.frame_num) / self.fps
        # Vis plots
        prob_plot_1 = plot.draw2D([x_axis[:i + 1]],
                                [self.total_eye1_prob],
                                order=[''],
                                xname='time',
                                yname='eye state',
                                params=params,
                                xlim=[0, max_X],
                                ylim=[-1, 2])

        prob_plot_2 = plot.draw2D([x_axis[:i + 1]],
                                [self.total_eye2_prob],
                                order=[''],
                                xname='time',
                                yname='eye state',
                                params=params,
                                xlim=[0, max_X],
                                ylim=[-1, 2])

        vis = np.concatenate([prob_plot_1, prob_plot_2], axis=1)
        scale = float(self.output_height) / vis.shape[0]
        # Resize plot size to same size with video
        vis = cv2.resize(vis, None, None, fx=scale, fy=scale)
        self.plot_vis_list.append(vis)
        return self.plot_vis_list
