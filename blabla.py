import cv2
import mmcv
import os
import numpy as np
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import torch

def get_gt(filename):
    #if filename ends with ng then it is a positive sample
    #if filename ends with ok then it is a negative sample
    if filename.endswith('ng.jpg'):
        return 1
    elif filename.endswith('ok.jpg'):
        return 0

def get_tp_fp_tn_fn(pred, gt):
    if pred == 1 and gt == 1:
        return 1, 0, 0, 0
    elif pred == 1 and gt == 0:
        return 0, 1, 0, 0
    elif pred == 0 and gt == 0:
        return 0, 0, 1, 0
    elif pred == 0 and gt == 1:
        return 0, 0, 0, 1
    else:
        raise Exception('invalid input')

config_file = 'configs/je/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco_je.py'
test_path = '/home/yaniv.sheinfeld/data/TEST_SET_6.6.6/201/images'
im_paths = [os.path.join(test_path, im) for im in os.listdir(test_path)]
gts = [get_gt(im_path) for im_path in im_paths]
thresholds = np.arange(0.1, 1.0, 0.1)
#for each threshold, calculate Detection Rate and False Positive Rate on all images in test_path
with open('results.txt', 'w') as f:
    for epoch in range(1,31):
        checkpoint_file = f'work_dirs/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco_je/epoch_{epoch}.pth'
        try:
            del model
            del results
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except:
            print('no model or results')
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        results = inference_detector(model, im_paths)
        for threshold in thresholds:
            f.write(f'epoch: {epoch}, threshold: {threshold}')
            f.write('\n')
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i,result in enumerate(results):
                pred = any(result.pred_instances.scores > threshold)
                tp, fp, tn, fn = get_tp_fp_tn_fn(pred, gts[i])
                TP += tp
                FP += fp
                TN += tn
                FN += fn
            DR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            f.write(f'DR: {DR}\n')
            f.write(f'FPR: {FPR}\n')
            f.write('')
            if DR < 0.95:
                break
        
