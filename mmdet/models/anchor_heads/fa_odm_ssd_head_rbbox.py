import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.ops import DeformConv
from mmdet.core import (AnchorGenerator, rotation_anchor_target_rbbox, weighted_smoothl1,
                        multi_apply, delta2dbbox, delta2dbbox_v3, delta2bbox, multiclass_nms_rbbox, multiclass_nms)
from .anchor_head_rbbox import AnchorHeadRbbox
from mmdet.core.bbox.transforms_rbbox import hbb2obb_v2
from ..registry import HEADS


# TODO: add loss evaluator for SSD
@HEADS.register_module
class FAODMSSDHeadRbbox(AnchorHeadRbbox):

    def __init__(self,
                 input_size=300,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_strides=(8, 16, 32, 64, 100, 300),
                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 with_module=False,
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)):
        super(AnchorHeadRbbox, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        self.with_module = with_module
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        reg_convs = []
        cls_convs = []
        off_sets_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                DeformConv(
                    in_channels[i],
                    num_anchors[i] * 5,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                DeformConv(
                    in_channels[i],
                    num_anchors[i] * num_classes,
                    kernel_size=3,
                    padding=1))
            off_sets_convs.append(
                nn.Conv2d(
                    num_anchors[i] * 2, 
                    2 * 3 * 3, 
                    1, 
                    bias=False))    
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)
        self.off_sets_convs = nn.ModuleList(off_sets_convs)

        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            stride = anchor_strides[k]
            ctr = ((stride - 1) / 2., (stride - 1) / 2.)
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            ratios = [1.]
            for r in anchor_ratios[k]:
                ratios += [1 / r, r]  # 4 or 6 ratio
            anchor_generator = AnchorGenerator(
                base_size, scales, ratios, scale_major=False, ctr=ctr)
            indices = list(range(len(ratios)))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(
                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats, arm_outs):
        cls_scores = []
        bbox_preds = []
        arm_cls_scores, arm_bbox_preds = arm_outs

        for feat, arm_bbox_pred, reg_conv, cls_conv, off_sets_conv in zip(feats, arm_bbox_preds,
                                                self.reg_convs, self.cls_convs, self.off_sets_convs):
                

     
                N, _, H, W = arm_bbox_pred.size()
                arm_bbox_pred = arm_bbox_pred.reshape(N,-1, 5, H, W)
                arm_ctr_off_sets = arm_bbox_pred[:, :, 0:2, :, :]
                arm_ctr_off_sets = arm_ctr_off_sets.reshape(N, -1, H, W)

                off_sets = off_sets_conv(arm_ctr_off_sets)
                cls_scores.append(cls_conv(feat, off_sets))
                bbox_preds.append(reg_conv(feat, off_sets))

        return cls_scores, bbox_preds

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        loss_bbox = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def loss(self,
             arm_cls_scores,
             arm_bbox_preds,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_mask,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        
        num_imgs = len(img_metas)
        _, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        arm_anchors_list = self.get_arm_bboxes(arm_bbox_preds, img_metas)

        cls_reg_targets = rotation_anchor_target_rbbox(
            arm_anchors_list,
            valid_flag_list,
            gt_bboxes,
            gt_mask,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            sampling=False,
            unmap_outputs=False,
            with_module=self.with_module)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 5)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 5)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 5)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos,
            cfg=cfg)
        return dict(loss_odm_cls=losses_cls, loss_odm_bbox=losses_bbox)
   
    def get_arm_bboxes(self, bbox_preds, img_metas, rescale=False):

        num_levels = len(bbox_preds)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(bbox_preds[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_arm_bboxes_single(bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, rescale)

            result_list.append(proposals)
        return result_list

    def get_arm_bboxes_single(self,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          rescale=False):
        assert len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []

        for bbox_pred, anchors in zip(bbox_preds, mlvl_anchors):
  
            # print(bbox_pred.size())
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            rbbox_ex_anchors = hbb2obb_v2(anchors)
            if self.with_module:
                bboxes = delta2dbbox(rbbox_ex_anchors, bbox_pred, self.target_means,
                                     self.target_stds, img_shape)
            else:
                bboxes = delta2dbbox_v3(rbbox_ex_anchors, bbox_pred, self.target_means,
                                     self.target_stds, img_shape)

            mlvl_bboxes.append(bboxes)

        return mlvl_bboxes
   
    def get_bboxes(self, arm_cls_scores, arm_bbox_preds, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_rbbox_anchors = self.get_arm_bboxes(arm_bbox_preds, img_metas)


        
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            # print('imgId', img_id)
            # print(len(mlvl_anchors[img_id]))
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_rbbox_anchors[img_id], img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_rbbox_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):

        assert len(cls_scores) == len(bbox_preds) == len(mlvl_rbbox_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, rbbox_anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_rbbox_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                rbbox_anchors = rbbox_anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = delta2dbbox_v3(rbbox_anchors, bbox_pred, self.target_means,
                                     self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        # import pdb
        # print('in anchor head get_bboxes_single')
        # pdb.set_trace()
        if rescale:
            mlvl_bboxes[:, :4] /= mlvl_bboxes[:, :4].new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
 

        det_bboxes, det_labels = multiclass_nms_rbbox(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
