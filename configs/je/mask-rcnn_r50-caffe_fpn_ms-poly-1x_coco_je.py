_base_ = "../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
data_root = "/home/yaniv.sheinfeld/data/je/"
metainfo = {
    "classes": ("chip", "crack", "grinding", "contamination", "dent", "sticky"),
    "palette": [
        (220, 20, 60),
    ]
    * 6,
}
optim_wrapper = dict(
    type='OptimWrapper',
    accumulative_counts=8,
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001))

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=6), mask_head=dict(num_classes=6)),
    data_preprocessor=dict(
        mean=[64.34, 64.34, 64.34], std=[42.92, 42.92, 42.92], bgr_to_rgb=False
    ),
)

train_pipeline = [
    dict(
        type="RandomChoiceResize",
        scales=[
            (1792, 1532),
            (2223, 1152),
            (2103, 991),
            (1824, 1563),
            (2236, 520),
            (2275, 571),
            (2093,1039),
            (2199,1051),
            (2034,1737)
        ],
        keep_ratio=True,
    ),
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        serialize_data=False,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        test_mode=True,))

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=f"{data_root}annotations/instances_train2017.json"
)


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=30)

# optimizer

# optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     accumulative_counts=8,
#     optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001))

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends,save_dir='tensorboard_logs' ,name='visualizer')