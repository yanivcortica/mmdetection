from mmengine.config import Config
config = '/home/yaniv.sheinfeld/repo/mmdetection/configs/je/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco_je.py'
cfg = Config.fromfile(config)
print(cfg.pretty_text)