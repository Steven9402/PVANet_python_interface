#!/usr/bin/env python
#-- coding:utf-8 --
import os.path as osp
import os
import cv2
import time
import copy
import numpy as np
from pascal_voc_io import  *

import _init_paths
import PvaDetector

imgdir = '/media/NEWDATA/newdata/Lishui/dj/lishui_lukou_pics/33.241.138.100'
res_xml_dir='/media/NEWDATA/newdata/Lishui/dj/lishui_lukou_det_xmls/33.241.138.100'
XML_EXT='.xml'
if __name__ == '__main__':
    prototxt_path = '/home/cuizhou/myGitRepositories/pvanet_python_interface/pva-faster-rcnn/models/all/all_comp.pt'
    model_path = '/home/cuizhou/myGitRepositories/pvanet_python_interface/pva-faster-rcnn/models/all/all_comp.model'
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    NMS_THRESH = 0.3
    CONF_THRESH = 0.7
    pva_detector = PvaDetector.PvaDetector(prototxt_path, model_path, CLASSES, NMS_THRESH, CONF_THRESH)

    for img_name in os.listdir(imgdir):
        srcImage = cv2.imread(osp.join(imgdir,img_name))
        detections = pva_detector.detect(srcImage)

        tmp = PascalVocWriter(res_xml_dir, img_name[:-4], srcImage.shape)
        for detection in detections:
            tmp.addBndBox(int(detection[1]), int(detection[2]), int(detection[3]), int(detection[4]), str(detection[0]))
        tmp.save(osp.join(res_xml_dir, img_name[:-4] + XML_EXT))
        print 'saving ',osp.join(res_xml_dir, img_name[:-4] + XML_EXT)
