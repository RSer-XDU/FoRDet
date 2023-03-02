import torch
import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import  bbox2result, build_assigner, build_sampler


@DETECTORS.register_module
class RefineStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 context= False):
        super(RefineStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_head = builder.build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.context = context

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(RefineStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_head.init_weights()



    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):


        x = self.backbone(img)
        if self.with_neck:
            trans_x = self.neck(x)

        losses = dict()

        # RPN forward and loss

        arm_outs = self.rpn_head(x)
        arm_loss_inputs = arm_outs + (gt_bboxes, None, img_meta,
                                        self.train_cfg.rpn)
        # print('--------', len(arm_loss_inputs))
        arm_losses = self.rpn_head.loss(
            *arm_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        losses.update(arm_losses)

        # bbox head forward and loss
 
        if self.context:
            outs = self.bbox_head(trans_x, arm_outs)
        else:
            outs = self.bbox_head(trans_x)
 

        # print(gt_bboxes_ignore)
        loss_inputs = arm_outs + outs + (gt_bboxes, gt_labels, img_meta, self.train_cfg.rcnn)

        odm_losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        losses.update(odm_losses)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        # assert self.with_bbox, "Bbox head must be implemented."


        x = self.backbone(img)
        if self.with_neck:
            trans_x = self.neck(x)


        # RPN forward and loss

        arm_outs = self.rpn_head(x)
        if self.context:
            outs = self.bbox_head(trans_x, arm_outs)
        else:
            outs = self.bbox_head(trans_x)

        bbox_inputs = arm_outs + outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]


