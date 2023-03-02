import torch
import numpy as np

from ..bbox import assign_and_sample, build_assigner, \
    PseudoRbboxSampler, bbox2delta, dbbox2delta, dbbox2delta_v3
from ..utils import multi_apply
from mmdet.core.bbox.transforms_rbbox import gt_mask_bp_obbs
# from ..bbox.geometry import bbox_overlaps
from DOTA_devkit_master.poly_nms_gpu.poly_overlaps import poly_overlaps
import numpy as np
def RotBox2Polys(dboxes):
    """
    :param dboxes: (x_ctr, y_ctr, w, h, angle)
        (numboxes, 5)
    :return: quadranlges:
        (numboxes, 8)
    """
    cs = np.cos(dboxes[:, 4])
    ss = np.sin(dboxes[:, 4])
    w = dboxes[:, 2] - 1
    h = dboxes[:, 3] - 1

    ## change the order to be the initial definition
    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    x1 = x1[:, np.newaxis]
    y1 = y1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    y2 = y2[:, np.newaxis]
    x3 = x3[:, np.newaxis]
    y3 = y3[:, np.newaxis]
    x4 = x4[:, np.newaxis]
    y4 = y4[:, np.newaxis]

    polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)
    return polys
def soft_rotation_anchor_target_rbbox(anchor_list,
                  cls_score_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  gt_masks_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  iter_factor=1.0,
                  sampling=True,
                  unmap_outputs=True,
                  with_module=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs == len(cls_score_list)

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])==len(cls_score_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
        cls_score_list[i] = torch.cat(cls_score_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         soft_rotation_anchor_target_rbbox_single,
         anchor_list,
         cls_score_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_masks_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         iter_factor=iter_factor,
         sampling=sampling,
         unmap_outputs=unmap_outputs,
         with_module=with_module)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def soft_rotation_anchor_target_rbbox_single(flat_anchors,
                         cls_scores,
                         valid_flags,
                         gt_bboxes,
                         gt_masks,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         iter_factor=1,
                         sampling=True,
                         unmap_outputs=True,                         
                         with_module=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]
    gt_obbs = gt_mask_bp_obbs(gt_masks, with_module)
    gt_obbs_ts = torch.from_numpy(gt_obbs)

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_obbs_ts, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_obbs_ts,
                                             gt_bboxes_ignore, gt_labels)
 
        bbox_sampler = PseudoRbboxSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_obbs_ts)

    num_valid_anchors = anchors.shape[0]
    # anchors shape, [num_anchors, 4]
    # bbox_targets = torch.zeros_like(anchors)
    # bbox_weights = torch.zeros_like(anchors)
    bbox_targets =  torch.zeros(num_valid_anchors, 5).to(anchors.device)
    bbox_weights = torch.zeros(num_valid_anchors, 5).to(anchors.device)

    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    # import pdb
    # print('in anchor target')
    # pdb.set_trace()
    pos_gt_rbboxes = sampling_result.pos_gt_bboxes.to(sampling_result.pos_bboxes.device)
    if len(pos_inds) > 0:
        # pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
        #                               sampling_result.pos_gt_bboxes,
        #                               target_means, target_stds)
        # if hbb_trans == 'hbb2obb':
        #     pos_ext_bboxes = hbb2obb(sampling_result.pos_bboxes)
        # elif hbb_trans == 'hbbpolyobb':
        #     pos_ext_bboxes = hbbpolyobb(sampling_result.pos_bboxes)
        # elif hbb_trans == 'hbb2obb_v2':
        #     pos_ext_bboxes = hbb2obb_v2(sampling_result.pos_bboxes)
        # else:
        #     print('no such hbb2obb trans function')
        #     raise Exception
        pos_rbboxes = sampling_result.pos_bboxes
        box_device = pos_rbboxes.device
        # print(pos_rbboxes.dtype)
        # print(pos_gt_rbboxes.dtype)
        pos_gt_rbboxes = pos_gt_rbboxes.float()
        pos_bbox_targets = dbbox2delta_v3(pos_rbboxes, pos_gt_rbboxes, target_means,
                                      target_stds)

        
        pos_gt_rbboxes_np = pos_gt_rbboxes.cpu().numpy()
        pos_rbboxes_np = pos_rbboxes.cpu().numpy()
        pos_gt_rbboxes_np = pos_gt_rbboxes_np.astype(np.float32)
        pos_rbboxes_np =pos_rbboxes_np.astype(np.float32)
        pos_gt_rbboxes_np[:, 2:4] =  pos_gt_rbboxes_np[:, 2:4] - 1
        pos_rbboxes_np[:, 2:4] =  pos_rbboxes_np[:, 2:4] - 1
        # # TODO poly overlaps
        pos_overlaps_np = poly_overlaps(pos_gt_rbboxes_np, pos_rbboxes_np)
        pos_overlaps_np = pos_overlaps_np.astype(np.float32)
        pos_overlaps = torch.from_numpy(pos_overlaps_np).to(box_device)
        torch.cuda.empty_cache()

        # pos_overlaps = bbox_overlaps(sampling_result.pos_gt_bboxes, sampling_result.pos_bboxes)
        # print('pos_gt_rbboxes_np', pos_gt_rbboxes_np)
        # print('pos_rbboxes_np', pos_rbboxes_np)
        # overlaps_index = torch.LongTensor(torch.unsqueeze(torch.arange(len(pos_inds)), 0)) 
        # print(pos_overlaps.size())
        # print(overlaps_index.size(), overlaps_index)
        # print(pos_overlaps.gather(0, overlaps_index))

        max_pos_overlaps, _ = torch.max(pos_overlaps, dim=0)
        # print(pos_overlaps, pos_overlaps.size())
        pos_cls_scores = cls_scores[pos_inds, 1]
        # print(max_pos_overlaps)
        # print(pos_cls_scores, pos_cls_scores.size())
        # assert 2==1
        re_alpha = 0.75
                # no converage
                # loc_a = 1 / (1 - max_pos_overlaps)
                # cls_c = 1 / (1 - pos_cls_scores)
        # print(iter_factor)
        loc_a = torch.exp(iter_factor * 2 * max_pos_overlaps)
        cls_c = torch.exp(iter_factor * 2 * pos_cls_scores)
        re_weight = re_alpha * loc_a + (1-re_alpha) * cls_c

        print(pos_rbboxes.size())
        print(pos_rbboxes)
        pos_polys = RotBox2Polys(pos_rbboxes.cpu().numpy())
        print(pos_polys)
        print(pos_gt_rbboxes.size())
        print(pos_gt_rbboxes)
        pos_gt_polys = RotBox2Polys(pos_gt_rbboxes.cpu().numpy())
        pos_gt_poly = pos_gt_polys[0]
        print(pos_gt_poly)
        f = open('pos_anchors.txt', 'a')
        gt_line = str(pos_gt_poly[0]) + ' ' + str(pos_gt_poly[1]) + ' ' + str(pos_gt_poly[2]) + ' ' + str(pos_gt_poly[3]) + ' ' + str(pos_gt_poly[4]) + ' ' + str(pos_gt_poly[5]) + ' ' + str(pos_gt_poly[6]) + ' ' + str(pos_gt_poly[7])  + ' gt' + '\n' 
        f.writelines(gt_line)
        
        for pos_poly, re_weight_ in zip(pos_polys, re_weight):
            re_weight_ = re_weight_.cpu().numpy()
            pos_line = str(pos_poly[0]) + ' ' + str(pos_poly[1]) + ' ' + str(pos_poly[2]) + ' ' + str(pos_poly[3]) + ' ' + str(pos_poly[4]) + ' ' + str(pos_poly[5]) + ' ' + str(pos_poly[6]) + ' ' + str(pos_poly[7])  + ' ' + str(re_weight_) + ' pos' + '\n'            
            f.writelines(pos_line)           
        # assert 2==1


        # print(re_weight, re_weight.size())
        bboxe_re_weight = torch.reshape(re_weight, (re_weight.size(0), 1)).repeat(1,5)
        # print(bboxe_re_weight[:, 0], bboxe_re_weight.size())

        # print(pos_inds.size())
        # assert 2==1

        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = bboxe_re_weight
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = re_weight
        else:
            label_weights[pos_inds] = re_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0
    # print(bbox_weights[pos_inds, :])
    # print(label_weights[pos_inds])
    # assert 2==1
    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    print(labels)
    print(labels.size())
    assert 2==1
    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
