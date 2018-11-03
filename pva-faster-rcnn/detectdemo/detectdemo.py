#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import PvaDetector
import cv2
import os
import os.path as osp
import numpy as np

# 下面两种读取方式都可行
# cls_path = osp.join('/home/cuizhou/projects/KaiKouXiao/mytrainedmodels/classes_name.txt')
# f = open(cls_path)
# class_name = ['__background__']
# while 1:
#     name = f.readline()
#     name = name.strip()
#     class_name.append(name)
#     if not name:
#         break
#
# class_name.pop()
# CLASSES = tuple(class_name)

def detectSingleImage(pva_detector):
    img_path = '/home/cuizhou/myGitRepositories/python_pva_interface/pva-faster-rcnn/data/demo/004545.jpg'
    image = cv2.imread(img_path)
    if type(image)!=np.ndarray:
        print 'empty image: ',img_path
    else:
        detections = pva_detector.detect(image)
        for detection in detections:
            cv2.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])),
                          (0, 255, 0), 1)
            cv2.putText(image, detection[0], (int(detection[1]), int(detection[2])), 0, 0.5, (0, 0, 255), 1)
        cv2.imshow('image',image)
        cv2.waitKey(0)

def detectImagesinFolder(pva_detector):
    folder_path = '/home/cuizhou/myGitRepositories/python_pva_interface/pva-faster-rcnn/data/demo'
    for name in os.listdir(folder_path):
        img_path = osp.join(folder_path,name)
        image = cv2.imread(img_path)
        if type(image) != np.ndarray:
            print 'empty image: ', img_path
        else:
            detections = pva_detector.detect(image)
            for detection in detections:
                cv2.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])),
                              (0, 255, 0), 1)
                cv2.putText(image, detection[0], (int(detection[1]), int(detection[2])), 0, 0.5, (0, 0, 255), 1)
            cv2.imshow('image', image)
            cv2.waitKey(0)

if __name__ == '__main__':

    prototxt_path = '/home/cuizhou/myGitRepositories/python_pva_interface/pva-faster-rcnn/models/all/all_comp.pt'
    model_path = '/home/cuizhou/myGitRepositories/python_pva_interface/pva-faster-rcnn/models/all/all_comp.model'
    CLASSES = ('__background__',
               'aeroplane','bicycle','bird','boat','bottle',
               'bus','car','cat','chair','cow','diningtable',
               'dog','horse','motorbike','person','pottedplant',
               'sheep','sofa','train','tvmonitor')

    NMS_THRESH = 0.3
    CONF_THRESH = 0.7
    pva_detector = PvaDetector.PvaDetector(prototxt_path, model_path, CLASSES,NMS_THRESH,CONF_THRESH)

    #detectSingleImage(pva_detector)
    detectImagesinFolder(pva_detector)
