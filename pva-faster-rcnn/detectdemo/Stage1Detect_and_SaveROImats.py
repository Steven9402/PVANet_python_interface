#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

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


# Odometer
cfg_file = osp.join('/home/cuizhou/projects/KaiKouXiao/mytrainedmodels/test.yml')
prototxt_path = osp.join('/home/cuizhou/projects/KaiKouXiao/mytrainedmodels/test.prototxt')
caffemodel_path = osp.join('/home/cuizhou/projects/KaiKouXiao/mytrainedmodels/test.caffemodel')
cls_path = osp.join('/home/cuizhou/projects/KaiKouXiao/mytrainedmodels/classes_name.txt')


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

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

        #cv2.imwrite()

    plt.imshow(im)
    plt.show()


def delete_box_iou(old_detections, thresh=0.4):
    new_detections = copy.copy(old_detections)
    index = []
    # ioulist = []
    for i in range(len(new_detections)):  # 0 -- len-1
        for j in range(i + 1, len(new_detections)):
            iou = getIOU(new_detections[i][1:5], new_detections[j][1:5])
            if iou >= thresh:
                # ioulist.append(iou)
                if new_detections[i][5] >= new_detections[j][5]:
                    index.append(j)
                else:
                    index.append(i)
    output = []
    for idx, detec in enumerate(new_detections):
        flag = 0
        for i in index:
            if idx == i:
                flag = 1
        if flag == 0:
            output.append(detec)

    for idx in index:
        new_detections[idx]

    return output


def get_detections(class_name, dets, thresh=0.5):
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
def demo2(net, im, _t, CLASSES, iou_thresh=0.4):
    """Detect object classes in an image using pre-computed object proposals."""
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, _t)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    # CONF_THRESH = 0.8
    CONF_THRESH = 0.4
    NMS_THRESH = 0.3
    detectrions_result = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        detections = get_detections(cls, dets, thresh=CONF_THRESH)

        # detections = get_detections(cls_ind, dets, thresh=CONF_THRESH)
        if len(detections) == 0:
            continue
        else:
            #print detections
            new_detections = delete_box_iou(detections, iou_thresh)
            #print new_detections
            detectrions_result.extend(new_detections)
    return detectrions_result

def detect(net, img, CLASSES, iou_thresh):
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    detections = demo2(net, img, _t, CLASSES, iou_thresh)
    return detections



if __name__ == '__main__':

    '''
    检测+裁剪保存
    step1：test_dir
    step2：save_path
    '''

    #test_dir = osp.join('/home/cuizhou/projects/KaiKouXiao/test_ws/stage1_result/qiepian_positive_high_resolution_python')
    test_dir = osp.join('/home/cuizhou/projects/KaiKouXiao/originaldata&&annotation/Negative')
    save_path = osp.join('/home/cuizhou/projects/KaiKouXiao/test_ws/stage1_result/part_roi_mats_for_2nd_from_negative')

    f = open(cls_path)
    class_name = ['__background__']
    while 1:
        name = f.readline()
        name = name.strip()
        class_name.append(name)
        if not name:
            break

    class_name.pop()
    CLASSES = tuple(class_name)


    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALE_MULTIPLE_OF = 32


    prototxt = prototxt_path
    caffemodel = caffemodel_path

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    #caffe.set_mode_cpu()

    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)


    test_imgs = os.listdir(test_dir)

    for img_path in test_imgs:

        image = cv2.imread(osp.join(test_dir, img_path))
        #image=makeBorder.mkBorder(image)
        dstImage = copy.deepcopy(image)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(img_path)

        iou_thresh=0.4
        detections = detect(net, image, CLASSES, iou_thresh)

        partidx = 0
        for detection in detections:
            # cv2.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])),
            #               (0, 255, 0), 1)

            newmat = image[int(detection[2]):int(detection[4]),int(detection[1]):int(detection[3])]
            cv2.imwrite(osp.join(save_path, img_path[0:len(img_path)-4]+'_'+str(partidx)+'.jpg'), newmat)

            partidx+=1


