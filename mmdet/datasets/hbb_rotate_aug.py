import mmcv
import numpy as np
from numpy import random
from mmdet.core import poly2bbox
from mmdet.core.evaluation.bbox_overlaps import  bbox_overlaps

import cv2
from functools import partial
import copy

def TupleBox2Poly(box):
    # print(box.dtype)
    outpoly = [box[0], box[1],
                       box[2], box[1],
                       box[2], box[3],
                       box[0], box[3]
                       ]
    return outpoly
def box2poly_single(boxes_list):
    """

    :param binary_mask:
    :return:
    """
    # try:
    poly = TupleBox2Poly(boxes_list)

    return poly

def box2poly(boxes_list):

    polys = map(box2poly_single, boxes_list)
    return list(polys)


def rotate_poly_single(h, w, new_h, new_w, rotate_matrix_T, poly):
    poly[::2] = poly[::2] - (w - 1) * 0.5
    poly[1::2] = poly[1::2] - (h - 1) * 0.5
    coords = poly.reshape(4, 2)
    new_coords = np.matmul(coords,  rotate_matrix_T) + np.array([(new_w - 1) * 0.5, (new_h - 1) * 0.5])
    rotated_polys = new_coords.reshape(-1, ).tolist()

    return rotated_polys

# TODO: refactor the single - map to whole numpy computation
def rotate_poly(h, w, new_h, new_w, rotate_matrix_T, polys):
    rotate_poly_fn = partial(rotate_poly_single, h, w, new_h, new_w, rotate_matrix_T)
    rotated_polys = list(map(rotate_poly_fn, polys))

    return rotated_polys

class HBBRotateAugmentation(object):
    """
    1. rotate image and polygons, transfer polygons to masks
    2. polygon 2 mask
    """
    def __init__(self,
                 # center=None,
                 CLASSES=None,
                 scale=1.0,
                 border_value=0,
                 auto_bound=True,
                 rotate_range=(-180, 180),
                 small_filter=4):
        self.CLASSES = CLASSES
        self.scale = scale
        self.border_value = border_value
        self.auto_bound = auto_bound
        self.rotate_range = rotate_range
        self.small_filter = small_filter
        # self.center = center

    def __call__(self, img, boxes, labels, filename):
        angle = np.random.rand() * (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0]
        discrete_range = [90, 180, -90, -180]
        for label in labels:
            # print('label: ', label)
            cls = self.CLASSES[label-1]
            # print('cls: ', cls)
            if (cls == 'storage-tank') or (cls == 'roundabout') or (cls == 'airport'):
                random.shuffle(discrete_range)
                angle = discrete_range[0]
                break

        # rotate image, copy from mmcv.imrotate

        h, w = img.shape[:2]
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        # print('len boxes: ', len(boxes))
        # print('len masks: ', len(masks))
        # print('len labels: ', len(labels))
        matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        matrix_T = copy.deepcopy(matrix[:2, :2]).T
        if self.auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated_img = cv2.warpAffine(img, matrix, (w, h), borderValue=self.border_value)

        polys = box2poly(boxes)

        rotated_polys = rotate_poly(img.shape[0], img.shape[1], h, w, matrix_T, np.array(polys))

        rotated_polys_np = np.array(rotated_polys)
        # add dimension in poly2mask
        # print('rotated_polys_np: ', rotated_polys_np)
        rotated_boxes = poly2bbox(rotated_polys_np).astype(np.float32)

        # print('len rotated boxes: ', len(rotated_boxes))
        # print('len rotaed polys: ', len(rotated_polys))
        # print('-----ratation---')
        # print('before len labels: ', len(labels))

        # True rotated h, sqrt((x1-x2)^2 + (y1-y2)^2)
        rotated_h = np.sqrt(np.power(rotated_polys_np[:, 0] - rotated_polys_np[:, 2], 2)
                            + np.power(rotated_polys_np[:, 1] - rotated_polys_np[:, 3], 2) )
        # True rotated w, sqrt((x2 - x3)^2 + (y2 - y3)^2)
        rotated_w = np.sqrt(np.power(rotated_polys_np[:, 2] - rotated_polys_np[:, 4], 2)
                            + np.power(rotated_polys_np[:, 3] - rotated_polys_np[:, 5], 2) )
        min_w_h = np.minimum(rotated_h, rotated_w)
        keep_inds = (min_w_h * img.shape[0] / np.float32(h)) >= self.small_filter
        # print(keep_inds, len(keep_inds))
        if len(keep_inds) > 0:

            rotated_boxes = rotated_boxes.astype(np.float32)
            rotated_boxes = rotated_boxes[keep_inds].tolist()
            aug_labels = labels[keep_inds]


            if len(aug_labels) == 0:
                return img, boxes, labels
        else:
            return img, boxes, labels

        return rotated_img, rotated_boxes, aug_labels
    
