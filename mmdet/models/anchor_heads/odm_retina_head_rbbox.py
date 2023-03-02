import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

# from .anchor_head import AnchorHead
from mmdet.core import (AnchorGenerator, rotation_anchor_target_rbbox,
                        multi_apply, delta2dbbox, delta2dbbox_v3, delta2bbox, multiclass_nms_rbbox, multiclass_nms)
from .anchor_head_rbbox import AnchorHeadRbbox
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule
from mmdet.core.bbox.transforms_rbbox import hbb2obb_v2

@HEADS.register_module
class ODMRetinaHeadRbbox(AnchorHeadRbbox):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(ODMRetinaHeadRbbox, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.large_kernel_l1 = nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=(3, 1), padding=(1, 0))
        self.large_kernel_l2 = nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=(1, 3), padding=(0, 1))
        self.large_kernel_r1 = nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=(1, 3), padding=(0, 1))
        self.large_kernel_r2 = nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=(3, 1), padding=(1, 0))

        for i in range(self.stacked_convs):
            chn = self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        normal_init(self.large_kernel_l1, std=0.01)
        normal_init(self.large_kernel_l2, std=0.01)
        normal_init(self.large_kernel_r1, std=0.01)
        normal_init(self.large_kernel_r2, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        feat_l = self.large_kernel_l1(x)
        feat_l = self.large_kernel_l2(feat_l)

        feat_r = self.large_kernel_r1(x)
        feat_r = self.large_kernel_r2(feat_r)
        cls_feat = feat_r + feat_l
        reg_feat = feat_r + feat_l

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

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
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss(self,
             arm_cls_scores,
             arm_bbox_preds,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        _, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        arm_anchors_list = self.get_arm_bboxes(arm_bbox_preds, img_metas)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = rotation_anchor_target_rbbox(
            arm_anchors_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            with_module=self.with_module)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
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



    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
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
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
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
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            rbbox_ex_anchors = hbb2obb_v2(anchors)
            if self.with_module:
                bboxes = delta2dbbox(rbbox_ex_anchors, bbox_pred, self.target_means,
                                     self.target_stds, img_shape)
            else:
                bboxes = delta2dbbox_v3(rbbox_ex_anchors, bbox_pred, self.target_means,
                                     self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[:, :4] /= mlvl_bboxes[:, :4].new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
        #                                         cfg.score_thr, cfg.nms,
        #                                         cfg.max_per_img)
        det_bboxes, det_labels = multiclass_nms_rbbox(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
