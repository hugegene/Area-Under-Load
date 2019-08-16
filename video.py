# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:58:47 2019

@author: bdgecyt
"""

import sys
import time
import datetime
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", type=str, default="C:\\Users\\bdgecyt\\Desktop\\constructionimages\\Ch2_20190301124233.mp4", help="path to dataset")
#    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
#    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
#    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
#    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
#    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
#    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
#    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
#    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
#    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
 
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    
    print("\nPerforming drop line estimation:")
    prev_time = time.time()
    
    videofile = opt.video_file
    
    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    vp = (800,900)
    
    ob1 = 1000
    ob2 = 500
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
#        print(frame.shape[:,:])
        print(ret)
        
        if ret:
            vps = wrapper.dealAImage(frame,"data/result/",False,False,False)
            
#            for line in lines:
#                cv2.line(frame, line[0], line[1], (0, 0, 255), 2)
            
            for vp in vps:
                cv2.circle(frame, (int(vp[0]), int(vp[1])), 20, (0,255,0), 3)
                cv2.circle(frame, (ob1,ob2), 20, (0,255,0), 3)
                cv2.line(frame, (int(vp[0]), int(vp[1])), (ob1,ob2), (0,255,0), 2)
            frames += 1
            ob1 -= 1
            
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            continue
            
        else:
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            continue
        
    cv2.destroyAllWindows()
    cap.release()