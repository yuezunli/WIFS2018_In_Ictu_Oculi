"""
Proj: YZ_utils
Date: 8/2/18
Written by Yuezun Li
--------------------------
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import warnings
from py_utils.face_utils.umeyama import umeyama
from tqdm import tqdm


mean_face_x = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689])

mean_face_y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182])

landmarks_2D = np.stack([mean_face_x, mean_face_y], axis=1)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
# OVERLAY_POINTS = [
#     LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
#     NOSE_POINTS + MOUTH_POINTS,
# ]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def bur_size(landmarks):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    return blur_amount


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def cut_head(imgs, point):
    h, w = imgs[0].shape[:2]
    x1, y1 = np.min(point, axis=0)
    x2, y2 = np.max(point, axis=0)
    delta_x = (x2 - x1) / 8
    delta_y = (y2 - y1) / 5
    delta_x = np.random.randint(delta_x)
    delta_y = np.random.randint(delta_y)
    x1_ = np.int(np.maximum(0, x1 - delta_x))
    x2_ = np.int(np.minimum(w-1, x2 + delta_x))
    y1_ = np.int(np.maximum(0, y1 - delta_y))
    y2_ = np.int(np.minimum(h-1, y2 + delta_y * 0.5))
    imgs_new = []
    for i, im in enumerate(imgs):
        im = im[y1_:y2_, x1_:x2_, :]
        imgs_new.append(im)
    locs = [x1_, y1_, x2_, y2_]
    return imgs_new, locs


def get_2d_aligned_face(image, mat, size, padding=[0, 0]):
    mat = mat * size
    mat[0, 2] += padding[0]
    mat[1, 2] += padding[1]
    return cv2.warpAffine(image, mat, (size + 2 * padding[0], size + 2 * padding[1]))


def get_face_loc(im, face_detector):
    """ get face locations, color order of images is rgb """
    face_locs = []
    faces = face_detector(np.uint8(im))
    if faces is None or len(faces) == 0:
        return None
    else:
        faces_list = []
        for i, d in enumerate(faces):
            faces_list.append([d.left(), d.top(), d.right(), d.bottom()])
        return faces_list


def align(im, face_detector, lmark_predictor):
    # This version we handle all faces in view
    # channel order rgb
    im = np.uint8(im)
    faces = face_detector(im)
    face_list = []
    if faces is not None or len(faces) > 0:
        for pred in faces:
            points = shape_to_np(lmark_predictor(im, pred))
            trans_matrix = umeyama(points[17:], landmarks_2D, True)[0:2]
            face_list.append([trans_matrix, points])
    return face_list


def get_aligned_face_and_landmarks(im, face_cache, aligned_face_size = 256, padding=(0, 0)):
    """
    get all aligned faces and landmarks of all images
    :param imgs: origin images
    :param fa: face_alignment package
    :return:
    """
    aligned_cur_shapes = []
    aligned_cur_im = []
    for mat, points in face_cache:
        # Get transform matrix
        aligned_face = get_2d_aligned_face(im, mat, aligned_face_size, padding)
        # Mapping landmarks to aligned face
        pred_ = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        pred_ = np.transpose(pred_)
        mat = mat * aligned_face_size
        mat[0, 2] += padding[0]
        mat[1, 2] += padding[1]
        aligned_pred = np.dot(mat, pred_)
        aligned_pred = np.transpose(aligned_pred[:2, :])
        aligned_cur_shapes.append(aligned_pred)
        aligned_cur_im.append(aligned_face)

    return aligned_cur_im, aligned_cur_shapes


def crop_eye(img, points):
    eyes_list = []

    left_eye = points[36:42, :]
    right_eye = points[42:48, :]

    eyes = [left_eye, right_eye]
    for j in range(len(eyes)):
        lp = np.min(eyes[j][:, 0])
        rp = np.max(eyes[j][:, 0])
        tp = np.min(eyes[j][:, -1])
        bp = np.max(eyes[j][:, -1])

        w = rp - lp
        h = bp - tp

        lp_ = int(np.maximum(0, lp - 0.25 * w))
        rp_ = int(np.minimum(img.shape[1], rp + 0.25 * w))
        tp_ = int(np.maximum(0, tp - 1.75 * h))
        bp_ = int(np.minimum(img.shape[0], bp + 1.75 * h))

        eyes_list.append(img[tp_:bp_, lp_:rp_, :])
    return eyes_list
