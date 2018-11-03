#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import math
import copy

import os.path as osp



def getIOU(Reframe,GTframe):
    ''' Rect = [x1, y1, x2, y2] '''
    x1 = Reframe[0];
    y1 = Reframe[1];
    width1 = Reframe[2]-Reframe[0];
    height1 = Reframe[3]-Reframe[1];

    x2 = GTframe[0];
    y2 = GTframe[1];
    width2 = GTframe[2]-GTframe[0];
    height2 = GTframe[3]-GTframe[1];

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0:
        ratio = 0
    else:
        Area = width*height;
        Area1 = width1*height1;
        Area2 = width2*height2;
        #ratio = Area*1.0/(Area1+Area2-Area);
        ratio = Area*1.0/max(Area1,Area2);
    # return IOU
    return ratio

#
# def delete_box_iou(old_detections, thresh=0.4):
#     new_detections = copy.copy(old_detections)
#     index = []
#     # ioulist = []
#     for i in range(len(new_detections)):  # 0 -- len-1
#         for j in range(i + 1, len(new_detections)):
#             iou = getIOU(new_detections[i][1:5], new_detections[j][1:5])
#             if iou >= thresh:
#                 # ioulist.append(iou)
#                 if new_detections[i][5] >= new_detections[j][5]:
#                     index.append(j)
#                 else:
#                     index.append(i)
#     output = []
#     for idx, detec in enumerate(new_detections):
#         flag = 0
#         for i in index:
#             if idx == i:
#                 flag = 1
#         if flag == 0:
#             output.append(detec)
#
#     for idx in index:
#         new_detections[idx]
#
#     return output
#
# def demo2(net, im, _t, CLASSES, iou_thresh=0.4):
#     """Detect object classes in an image using pre-computed object proposals."""
#     timer = Timer()
#     timer.tic()
#     scores, boxes = im_detect(net, im, _t)
#     timer.toc()
#     print ('Detection took {:.3f}s for '
#            '{:d} object proposals').format(timer.total_time, boxes.shape[0])
#
#     # Visualize detections for each class
#     # CONF_THRESH = 0.8
#     CONF_THRESH = 0.4
#     NMS_THRESH = 0.3
#     detectrions_result = []
#     for cls_ind, cls in enumerate(CLASSES[1:]):
#         cls_ind += 1  # because we skipped background
#         cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
#         cls_scores = scores[:, cls_ind]
#         dets = np.hstack((cls_boxes,
#                           cls_scores[:, np.newaxis])).astype(np.float32)
#         keep = nms(dets, NMS_THRESH)
#         dets = dets[keep, :]
#
#         detections = get_detections(cls, dets, thresh=CONF_THRESH)
#
#         # detections = get_detections(cls_ind, dets, thresh=CONF_THRESH)
#         if len(detections) == 0:
#             continue
#         else:
#             #print detections
#             new_detections = delete_box_iou(detections, iou_thresh)
#             #print new_detections
#             detectrions_result.extend(new_detections)
#     return detectrions_result

class PvaDetector:
    'pva检测器'

    def __init__(self, _protxtpath, _modelpath, _CLASSES,_NMS_THRESH=0.3,_CONF_THRESH=0.7):
        self.protxtpath = _protxtpath
        self.modelpath = _modelpath
        self.CLASSES = _CLASSES
        self.NMS_THRESH = _NMS_THRESH
        self.CONF_THRESH = _CONF_THRESH

        # 加载网络
        #if caffemode == '0':
        #    caffe.set_mode_cpu()
        #else:

        caffe.set_mode_gpu()
        caffe.set_device(0)  # args.gpu_id)

        cfg.GPU_ID = 0  # args.gpu_id
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        cfg.TEST.SCALE_MULTIPLE_OF = 32

        self.net = caffe.Net(_protxtpath, _modelpath, caffe.TEST)


    def get_detections(self,class_name, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return []
        detections = []
        for i in inds:
            bbox1 = dets[i, :4]
            bbox = bbox1.tolist()
            score1 = dets[i, -1]
            score = score1.tolist()
            detection = [class_name, bbox[0], bbox[1], bbox[2], bbox[3], score]
            detections.append(detection)

        return detections


    def demo(self, im, _t):
        """Detect object classes in an image using pre-computed object proposals."""

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.net, im, _t)
        timer.toc()

        detectrions_result = []
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self.NMS_THRESH)
            dets = dets[keep, :]
            detections = self.get_detections(cls, dets, thresh=self.CONF_THRESH)
            # detections = get_detections(cls_ind, dets, thresh=CONF_THRESH)
            if len(detections) == 0:
                continue
            else:
                detectrions_result.extend(detections)
        return detectrions_result

    def detect(self,img):
        _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
        detections = self.demo(img, _t)
        return detections

