# model settings
input_size = 512
model = dict(
    type='RefineStageDetectorRbbox',
    pretrained='open-mmlab://vgg16_caffe',
    context=True,
    backbone=dict(
        type='SSDVGG',
        input_size=input_size,
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        l2_norm_scale=20),
    neck=dict(
        type='RefineFPN',
        in_channels=[512, 1024, 512, 256, 256],
        out_channels=256),
    rpn_head=dict(
        type='ARMSSDHeadRbbox',
        input_size=input_size,
        in_channels=(512, 1024, 512, 256, 256),
        num_classes=2,
        anchor_strides=(8, 16, 32, 64, 128),
        basesize_ratio_range=(0.1, 0.9),
        anchor_ratios=([2], [2], [2], [2], [2]),
        target_means=(.0, .0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
    rbbox_head=dict(
        type='FRODMSSDHeadRbbox',
        input_size=input_size,
        in_channels=(256, 256, 256, 256, 256),
        num_classes=16,
        anchor_strides=(8, 16, 32, 64, 128),
        basesize_ratio_range=(0.1, 0.9),
        anchor_ratios=([2], [2], [2], [2], [2]),
        target_means=(.0, .0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)))
cudnn_benchmark = True
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    rcnn = dict(
        assigner=dict(
            type='MaxIoUAssignerRbbox',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False)
    )
test_cfg = dict(
    nms=dict(type='py_cpu_nms_poly_fast', iou_thr=0.1),
    min_bbox_size=0,
    score_thr=0.05,
    max_per_img = 2000)
# model training and testing settings
# dataset settings
# dataset settings
dataset_type = 'DOTADataset'
data_root = '/media/xaserver/DATA/zty/FoRDet/DOTA/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'trainSplit1024_obb/DOTA_train1024.json',
            img_prefix=data_root + 'trainSplit1024_obb/images/',
            img_scale=(512, 512),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=True,
            with_crowd=True,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 1.2)),
                random_crop=dict(
                    min_ious=(0.7, 0.8, 0.9), min_crop_size=0.7)),
            rotate_aug=dict(border_value=0, small_filter=6),
            resize_keep_ratio=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'valSplit1024_obb/DOTA_val1024.json',
        img_prefix=data_root + 'valSplit1024_obb/images/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[8, 10])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'worker_dir/fr_8_refine_ssd512_obb'
load_from = None
resume_from = None
workflow = [('train', 1)]
