import os
import logging
import numpy as np

from PIL import Image
from torch_mtcnn import detect_faces


# setup logger
logger = logging.getLogger(os.path.basename(__file__))


def fetch_faces(image, return_landmarks=False):
    # standarize detector input
    if isinstance(image, (np.ndarray, np.generic)):
        image = Image.fromarray(image)

    # for some reason, detector randomly throws an error
    try:
        bboxes, landmarks = detect_faces(image)
    except ValueError:
        bboxes, landmarks = [], []

    # postprocess bounding bboxes
    if len(bboxes) > 0:
        scores = bboxes[:, -1]
        bboxes = bboxes[:, :-1].astype("int")
    else:
        scores = []

    return ([bboxes, scores], landmarks) if return_landmarks else bboxes, scores


def fetch_centroids(bboxes):
    if len(bboxes) == 0:
        return []
    return np.c_[
        (bboxes[:, 0] + bboxes[:, 2]) / 2,
        (bboxes[:, 1] + bboxes[:, 3]) / 2,
    ].astype("float")
