import mmcv
import numpy as np
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from pycocotools.coco import maskUtils
from functools import partial
import cv2

def filter (img, boxes, masks, labels, filename,method=None):
        # print('beofre', filename, len(masks), len(labels))
        # filter_polys = []
        # filter_labels = []
        filter_flag = 0
        # print('---mask extra')
        # print(masks.shape)
        for mask, label in zip(masks, labels):
            # print(mask.shape)
            try:
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                max_contour = max(contours, key=len)
                # rect = cv2.minAreaRect(max_contour)
                # poly = cv2.boxPoints(rect)
                # poly = TuplePoly2Poly(poly)
                # filter_polys.append(poly)
                # filter_labels.append(label)
            except:
                filter_flag = 1
                continue
        # print(filter_polys)
        # filter_labels = np.array(filter_labels)

        # filter_polys_np = np.array(filter_polys)
        # print('filter_poly', filename, filter_polys_np.shape)
        if filter_flag == 1:
            print('-------------------', method, filename)
            return img, np.zeros((0,4), dtype=np.float32).tolist(), [], np.array([], dtype=np.int64)
        else:
            # filter_masks = poly2mask(filter_polys_np[:, np.newaxis, :].tolist(), img.shape[0], img.shape[1])
            # print('rotated_masks: ', rotated_masks)
            # print('rotated masks sum: ', sum(sum(rotated_masks[0])))
            # filter_boxes = poly2bbox(filter_polys_np).astype(np.float32)
            # print('after', filename, len(filter_masks), len(filter_labels))

            return img, boxes, masks, labels



# def filter (img, boxes, masks, labels, filename,method=None):
#         # print('beofre', filename, len(masks), len(labels))
#         filter_polys = []
#         filter_labels = []
#         for mask, label in zip(masks, labels):
#             try:
#                 contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#                 max_contour = max(contours, key=len)
#                 rect = cv2.minAreaRect(max_contour)
#                 poly = cv2.boxPoints(rect)
#                 poly = TuplePoly2Poly(poly)
#                 filter_polys.append(poly)
#                 filter_labels.append(label)
#             except:
#                 continue
#         # print(filter_polys)
#         filter_labels = np.array(filter_labels)
#         filter_polys_np = np.array(filter_polys)
#         # print('filter_poly', filename, filter_polys_np.shape)
#         if len(filter_polys_np) == 0:
#             print(filter_polys_np)
#             print('-------------------', method, filename)
#             return img, np.zeros((0,4), dtype=np.float32).tolist(), [], np.array([], dtype=np.int64)
#         else:
#             filter_masks = poly2mask(filter_polys_np[:, np.newaxis, :].tolist(), img.shape[0], img.shape[1])
#             filter_boxes = poly2bbox(filter_polys_np).astype(np.float32)
#             return img, filter_boxes, filter_masks, filter_labels

def poly2mask_single(h, w, poly):
    # TODO: write test for poly2mask, using mask2poly convert mask to poly', compare poly with poly'
    # visualize the mask
    rles = maskUtils.frPyObjects(poly, h, w)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)

    return mask

def poly2mask(polys, h, w):
    poly2mask_fn = partial(poly2mask_single, h, w)
    masks = list(map(poly2mask_fn, polys))
    # TODO: check if len(masks) == 0
    return masks

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def mask2poly_single(binary_mask):
    """

    :param binary_mask:
    :return:
    """
    # try:
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=len)
    rect = cv2.minAreaRect(max_contour)
    poly = cv2.boxPoints(rect)
    poly = TuplePoly2Poly(poly)

    return poly

def mask2poly(binary_mask_list):
    polys = map(mask2poly_single, binary_mask_list)
    return list(polys)


def poly2bbox(polys):
    """
    without label
    :param polys: (x1, y1, ..., x4, y4) (n, 8)
    :return: boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    n = polys.shape[0]
    xs = np.reshape(polys, (n, 4, 2))[:, :, 0]
    ys = np.reshape(polys, (n, 4, 2))[:, :, 1]

    xmin = np.min(xs, axis=1)
    ymin = np.min(ys, axis=1)
    xmax = np.max(xs, axis=1)
    ymax = np.max(ys, axis=1)

    xmin = xmin[:, np.newaxis]
    ymin = ymin[:, np.newaxis]
    xmax = xmax[:, np.newaxis]
    ymax = ymax[:, np.newaxis]

    return np.concatenate((xmin, ymin, xmax, ymax), 1)

class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, masks, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, masks, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, masks, labels, filename):
        if random.randint(2):
            return img, boxes, masks, labels
        # print('len boxes: ', len(boxes))
        # print('---expand----------')
        # print('len labels: ', len(labels))
        # print('expand boxes', boxes[0])
        # print('mask_polys', mask_polys[0, :])
 
  
        h, w, c = img.shape
        
        # print(h,w,c)
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        # print(ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        # print('l,t', left, top)
        expand_img[top:top + h, left:left + w] = img

        orig_polys = mask2poly(masks)    
        mask_polys = np.copy(orig_polys)
        mask_polys += np.tile((left, top), 4)
        aug_masks = poly2mask(mask_polys[:, np.newaxis, :].tolist(), expand_img.shape[0], expand_img.shape[1])
        aug_boxes = poly2bbox(mask_polys).astype(np.float32)


        # True rotated h, sqrt((x1-x2)^2 + (y1-y2)^2)
        trans_h = np.sqrt(np.power(mask_polys[:, 0] - mask_polys[:, 2], 2)
                            + np.power(mask_polys[:, 1] - mask_polys[:, 3], 2) )
        # True rotated w, sqrt((x2 - x3)^2 + (y2 - y3)^2)
        trans_w = np.sqrt(np.power(mask_polys[:, 2] - mask_polys[:, 4], 2)
                            + np.power(mask_polys[:, 3] - mask_polys[:, 5], 2) )
        min_w_h = np.minimum(trans_h, trans_w)
        keep_inds = (min_w_h * img.shape[0] / np.float32(expand_img.shape[0])) >= 6
        # print(keep_inds, len(keep_inds))

        if len(keep_inds) > 0:
            aug_boxes = aug_boxes[keep_inds]
            aug_masks = np.array(aug_masks)[keep_inds]
            aug_labels = labels[keep_inds]
            # print('expand')
            # print('filename', filename, len(aug_boxes), len(aug_labels), len(aug_masks), len(keep_inds))


            # changes ZTY 20210725
            # expand_img, aug_boxes, aug_masks, aug_labels = filter(expand_img, aug_boxes, aug_masks, aug_labels, filename, 'expand')
            # changes ZTY 20210725

            
            if len(aug_masks) == 0:
                return img, boxes, masks, labels
            else:
                return expand_img, aug_boxes, aug_masks, aug_labels
        else:
            # print('no expand')
            return img, boxes, masks, labels
 



class RandomCrop(object):

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, masks, labels, filename):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, masks, labels

            min_iou = mode
            # print('---mode--', mode)
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                # new_h = random.uniform(self.min_crop_size * h, h)
                new_h = new_w


                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)


                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                
                
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)

                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                filter_boxes = boxes[mask]
                filter_labels = labels[mask]
                # print(type(labels))
                # print(type(masks))
                filter_masks = np.array(masks)[mask]

                # print('------crop------')
                crop_img = img[patch[1]:patch[3], patch[0]:patch[2]]
                orig_polys = mask2poly(filter_masks)    
                mask_polys = np.copy(orig_polys)

                mask_polys[:, 0:2:8] = np.clip(mask_polys[:, 0:2:8], a_min=patch[0], a_max=patch[2])
                mask_polys[:, 1:2:8] = np.clip(mask_polys[:, 1:2:8], a_min=patch[1], a_max=patch[3])
 
                mask_polys -= np.tile(patch[:2], 4)
                mask_polys = np.maximum(mask_polys, 0)

                aug_masks = poly2mask(mask_polys[:, np.newaxis, :].tolist(), crop_img.shape[0], crop_img.shape[1])
                # print('rotated_masks: ', rotated_masks)
                # print('rotated masks sum: ', sum(sum(rotated_masks[0])))
                aug_boxes = poly2bbox(mask_polys).astype(np.float32)

                # print('len rotated boxes: ', len(rotated_boxes))
                # print('len rotaed polys: ', len(rotated_polys))
                # print('len rotated_masks: ', len(rotated_masks))
                # print('before len labels: ', len(labels))

                # True rotated h, sqrt((x1-x2)^2 + (y1-y2)^2)
                trans_h = np.sqrt(np.power(mask_polys[:, 0] - mask_polys[:, 2], 2)
                                    + np.power(mask_polys[:, 1] - mask_polys[:, 3], 2) )
                # True rotated w, sqrt((x2 - x3)^2 + (y2 - y3)^2)
                trans_w = np.sqrt(np.power(mask_polys[:, 2] - mask_polys[:, 4], 2)
                                    + np.power(mask_polys[:, 3] - mask_polys[:, 5], 2) )
                min_w_h = np.minimum(trans_h, trans_w)
                keep_inds = (min_w_h * img.shape[0] / np.float32(crop_img.shape[0])) >= 6
                # print(keep_inds, len(keep_inds))

                if len(keep_inds) > 0:
                    aug_boxes = aug_boxes[keep_inds].tolist()
                    aug_masks = np.array(aug_masks)[keep_inds]
                    aug_labels = filter_labels[keep_inds]
                    # print('crop')
                    # print('filename', filename, len(boxes), len(aug_boxes), len(aug_labels), len(aug_masks), len(keep_inds))
                    crop_img, aug_boxes, aug_masks, aug_labels = filter(crop_img, aug_boxes, aug_masks, aug_labels, filename,'crop')
                    if len(aug_masks) == 0:
                        return img, boxes, masks, labels
                    else:
                        return crop_img, aug_boxes, aug_masks, aug_labels
                else:
                    return img, boxes, masks, labels
        




class MaskExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.photo_metric_distortion = (
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, boxes, masks, labels, filename):
        img = img.astype(np.float32)

        # if random.randint(2):
                # img, boxes, masks, labels = self.photo_metric_distortion(img, boxes, masks, labels)      
        for transform in self.transforms:
                img, boxes, masks, labels = transform(img, boxes, masks, labels, filename)
        
        return img, boxes, masks, labels



