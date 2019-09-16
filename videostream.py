# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:19:23 2019

@author: bdgecyt
"""
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from torchvision import transforms
import torch
from PoseMachine import DopeNetwork, padding, image_resize
import numpy as np
from vectorprojection import calibrateVP, calibratePlane
import matplotlib as plt
from functools import reduce
import operator
import math

def cornerdetection(load):
#        print("start corner inference")
    size = load.shape
    load = image_resize(load, width = 400)
    load = padding(load)
    load = load[:400,:400,:]
    
    data_transform = transforms.Compose([transforms.ToTensor()])
    
    inp = data_transform(load)
    inp = inp.view([1, 3, 400, 400])
 
    out = net(inp)[0].data.numpy().reshape(4, 50, 50)
    
#    predictedcoor = []
    globalcoor =[]
    
    for i in out:
#        print(i.shape)
        argmax = np.unravel_index(i.argmax(), i.shape)
#        print(argmax)
        argmax = np.array(argmax)/50
#        print(argmax)
#            print(argmax*size[:2])
#            print(argmax*size[:2] + np.array([y,x]))
        
#        print(argmax)
#        predictedcoor += [argmax]
        globalcoor +=[argmax*size[:2] + np.array([y,x])]
    
#    for i in predictedcoor:
#         print(i[1], i[0])
#         print(tuple([int(i[1]*resize[1]), int(i[0]*resize[0])]))
##        print(i[0]*400, i[1]*400)
##        print(i[1]*400)
#         cv2.circle(load, tuple([int(i[1]*resize[1]), int(i[0]*resize[0])]) ,3,(255,0,0),-1)
#    
#    for i in globalcoor:
#        print(i)
#        cv2.circle(load, tuple([np.int32(i[1]), np.int32(i[0])]), 3, (255,0,255), -1)  
    globalcoor = np.flip(globalcoor, axis=1)

#        cv2.circle(frame, tuple([x, y]) ,6,(255,255,0),-1)
    
#        for i in globalcoor:
#            cv2.circle(frame, tuple([np.int32(i[0]), np.int32(i[1])]), 3, (255,0,255), -1)  
    return globalcoor
        
#        cv2.imshow("image", frame)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()


if __name__ == '__main__' :
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default="data\\test2.mp4", type=str,
    	help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt",
    	help="OpenCV object tracker type")
    args = vars(ap.parse_args())
    
    print("calibrating plane and vanshing points")
    zVanish, xVanish, yVanish = calibrateVP()
    
    zVanish= np.array([850, 1395])
    
    vs = cv2.VideoCapture(args["video"])
    ret, frame = vs.read()
    plane1 = calibratePlane(zVanish)
    
    src = [[634, 479],[636, 368],[520, 365],[519, 480],[477, 480],[479, 366],[398, 401],[362, 481]]
    dst = [[959, 652],[1284, 412],[995, 269],[692, 414],[621, 352],[934, 228],[700, 190],[489, 218]]
    plane1.calibrate(dstimage= frame, gridsize =10, srcpts =src, dstpts = dst)
    
    print("load corner detection model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DopeNetwork(pretrained=False).to(device)
    net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(torch.load("C:\\Users\\bdgecyt\\Desktop\\poseparameters.pth", map_location=torch.device('cpu')))
    
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
     
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
    	tracker = cv2.Tracker_create(args["tracker"].upper())
     
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    else:
    	# initialize a dictionary that maps strings to their corresponding
    	# OpenCV object tracker implementations
    	OPENCV_OBJECT_TRACKERS = {
    		"csrt": cv2.TrackerCSRT_create,
    		"kcf": cv2.TrackerKCF_create,
    		"boosting": cv2.TrackerBoosting_create,
    		"mil": cv2.TrackerMIL_create,
    		"tld": cv2.TrackerTLD_create,
    		"medianflow": cv2.TrackerMedianFlow_create,
    		"mosse": cv2.TrackerMOSSE_create
    	}
     
    	# grab the appropriate object tracker using our dictionary of
    	# OpenCV object tracker objects
    	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
     
    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None
    
    # if a video path was not supplied, grab the reference to the web cam
    if not args.get("video", False):
    	print("[INFO] starting video stream...")
    	vs = VideoStream(src=0).start()
    	time.sleep(1.0)
     
    # otherwise, grab a reference to the video file
    else:
    	vs = cv2.VideoCapture(args["video"])
  
     
    # initialize the FPS throughput estimator
    fps = None
    n = 0
    # loop over frames from the video stream
    sec = 1/1
    count = sec
    while True:
#        vs.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) 
        # grab the current frame, then handle if we are using a
    	# VideoStream or VideoCapture object
#        print("new frame")
        n+=1
        frame = vs.read()
        
        
        frame = frame[1] if args.get("video", False) else frame
    	
    	# check to see if we have reached the end of the stream
        if frame is None:
            break
    	# resize the frame (so we can process it faster) and grab the
    	# frame dimensions
    #    frame = imutils.resize(frame, width=1000)
        (H, W) = frame.shape[:2]
    	
#        count = count + sec
    	# check to see if we are currently tracking an object
        if initBB is not None:
    		# grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
    		
    		# check to see if the tracking was a success
            if success:
                    
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

                load= frame[y:y+h,x:x+w,:]
#                plt.image.imsave('data\\crop\\test2_' + str(n) +'.jpg', load)
                
                glo = cornerdetection(load)
#                print("detection")
#                print(glo)
                center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), glo), [len(glo)] * 2))
                coords = np.array(sorted(glo, key=lambda coord: (360 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))
                
                c = 255
                b= 0
                for i in coords:
                    cv2.circle(frame, tuple([np.int32(i[0]), np.int32(i[1])]), 6, (c,b,0), -1)
                    cv2.line(frame, pt1=tuple(plane1.zVanish), pt2=tuple([np.int32(i[0]), np.int32(i[1])]), color=(0,255,255),thickness=2)
                    c-=50
                    b+=50
                area = plane1.detectFallArea(coords, frame)
                
                c = 255
                b= 0
                for i in area:
                    cv2.circle(frame, tuple(i), 8, (c,b,0), -1)
                    c-= 50
                    b+= 50
              
    		# update the FPS counter
            fps.update()
            fps.stop()
    		
    		# initialize the set of information we'll be displaying on
    		# the frame
            info = [
    			("Tracker", args["tracker"]),
    			("Success", "Yes" if success else "No"),
    			("FPS", "{:.2f}".format(fps.fps())),
    		]
    		
     
    		# loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
    				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    			
    	# show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
    	# if the 's' key is selected, we are going to "select" a bounding
    	# box to track
        if key == ord("s"):
    	
    		# select the bounding box of the object we want to track (make
    		# sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
    			showCrosshair=True)
    		
     
    		# start OpenCV object tracker using the supplied bounding box
    		# coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()
    		
        	# if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
    	
    # if we are using a webcam, release the pointer
    if not args.get("video", False):
    	vs.stop()
     
    # otherwise, release the file pointer
    else:
    	vs.release()
     
    # close all windows
    cv2.destroyAllWindows()
    
#    plane1.householdshelterlength
#
#    plane1.householdshelterwidth
    

    
#    print("-----start area under load----------")
#   
#    
#    for i in np.array(globalcoor[:3], np.int64):
#        cv2.line(frame, pt1=tuple(zVanish), pt2=tuple(i),color=(0,255,255),thickness=2)
##    refcoor = CoordinateStore(im_dst)
##    cv2.namedWindow('image')
##    cv2.setMouseCallback('image', refcoor.select_point)
##    while(1):
##        if len(refcoor.points) != 0:
##            for i in refcoor.points:
##                cv2.line(im_dst,pt1=tuple(zVanish),pt2=tuple(i),color=(0,255,255),thickness=2)
##
##        cv2.imshow('image', im_dst)
##        k = cv2.waitKey(1) & 0xFF
##        if k == 27:
##            break
##    cv2.destroyAllWindows()
##    objRefpts = np.array(refcoor.points, np.int64)
#    plane1.detectFallArea(np.array(globalcoor[:3], np.int64), frame)

