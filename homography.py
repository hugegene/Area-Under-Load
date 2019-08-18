# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:40:22 2019

@author: bdgecyt
"""
from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sucestest import calibrateframe
import matplotlib.path as mpltPath
import timeit

class CoordinateStore:
    def __init__(self, img):
        self.points = []
        self.img = img
        self.line =[]
        self.lines = []

    def select_point(self,event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                print(x,y)
                cv2.circle(self.img,(x,y),3,(255,0,0),-1)
                self.points.append((x,y))
                
    def draw_line(self,event,x,y,flags,param):
#            if event == cv2.EVENT_LBUTTONDBLCLK:
#                print(x,y)
#                cv2.circle(self.img,(x,y),3,(255,0,0),-1)
#                self.points.append((x,y))
            if event == cv2.EVENT_LBUTTONDOWN:
                print('Start Mouse Position: '+str(x)+', '+str(y))
                self.line = []
                self.line.append([x,y])
                print(self.line)
#                cv2.line(self.img, pt1=(self.line[0]),pt2=(x,y),color=(255,255,0),thickness=2)
                
            elif event == cv2.EVENT_LBUTTONUP:
                print('End Mouse Position: '+str(x)+', '+str(y))
                self.line.append([x,y])
                self.lines.append(self.line)
                print("line created")
                print(self.lines)
                
def calibrateVP():
    im_dst = cv2.imread("data\\blkcamera.jpg")
    zVanish, xVanish, yVanish = calibrateframe(im_dst)
    zVanish = np.array((int(zVanish[0]), int(zVanish[1])))
    return zVanish

class calibratePlane():
    def __init__(self, zVanish):
        self.rowlines = []
        self.rowsmask = []
        self.householdshelterlength =[]
        self.dst1 = []
        self.zVanish = zVanish

    def calibrate(self, dstimage = "data\\blkcamera.jpg", srcimage = "data\\blkplan2.jpg"):    
        im_dst = cv2.imread(dstimage)
        #draw building plane   
        im_src = cv2.imread(srcimage)
    
        polycoor = CoordinateStore(im_src)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', polycoor.select_point)
        while(1):
            if len(polycoor.points) >2:
                im_src = cv2.imread("data\\blkplan2.jpg")
                cv2.polylines(im_src, [np.array(polycoor.points)], True, (0,255,0), thickness=3)
    #            cv2.line(im_dst,pt1=polycoor.points[0],pt2=polycoor.points[1],color=(0,255,255),thickness=2)
    #        cv2.putText(im_src,"Select border of horizontal plane", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            cv2.imshow('image', im_src)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        
        # Create a source image, a window and bind the function to window    
        rows = np.arange(0, im_src.shape[0], 4)
        cols = np.arange(0, im_src.shape[1], 4)
        points = [[i, j] for j in rows for i in cols]
#        print(len(points))
    
        housholdshelter = [[1146, 395], [1203, 395]]
        householdshelterlength = np.argmin(abs(cols -housholdshelter[1][0]))-np.argmin(abs(cols -housholdshelter[0][0]))
        
        path = mpltPath.Path(polycoor.points)
        inside2 = path.contains_points(points)
        insidepts = [points[i] for i in range(len(points)) if inside2[i] == True]
    
        rowsmask =[]
        previous = -1
        a= -1
        for i in insidepts:
            if i[1] != previous:
                previous = i[1]
                a +=1
            rowsmask.append(a)
    
    
        plt.imshow(im_src)
        plt.scatter([i[0] for i in insidepts], [i[1] for i in insidepts], color = 'blue')
        plt.show()
        
        coordinateStore1 = CoordinateStore(im_src)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',coordinateStore1.select_point)
        
        while(1):
            cv2.imshow('image',im_src)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        
        coordinateStore2 = CoordinateStore(im_dst)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',coordinateStore2.select_point)
        
        while(1):
            cv2.imshow('image', im_dst)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        
        # Calculate Homography
        h, status = cv2.findHomography(np.array(coordinateStore1.points), np.array(coordinateStore2.points))
        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
         
        # Display warp plan
        cv2.imshow("Warped Source Image", im_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
     
        pts = np.array(insidepts, np.float32)
        pts1 = pts.reshape(-1,1,2).astype(np.float32)
        dst1 = cv2.perspectiveTransform(pts1, h)
        dst1 = dst1.reshape([dst1.shape[0],dst1.shape[2]])
    #    print(dst1.shape)
        
        for i in dst1:
            cv2.circle(im_dst,(int(i[0]),int(i[1])),3,(255,0,0),-1)
        cv2.imshow("image", im_dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        uniquerows = np.unique(rowsmask)
        rowend1= np.array([dst1[rowsmask==i][0] for i in uniquerows])
        rowend2= np.array([dst1[rowsmask==i][-1] for i in uniquerows])
        rowend1 = cv2.convertPointsToHomogeneous(rowend1) 
        rowend2 = cv2.convertPointsToHomogeneous(rowend2) 
        rowlines = np.cross(rowend1, rowend2)
        
        self.rowlines = rowlines
        self.rowsmask = rowsmask
        self.householdshelterlength = householdshelterlength
        self.dst1 = dst1
    
#    return rowlines, rowsmask, householdshelterlength, dst1
    def detectFallArea(self, objRefpts, im_dst):
        
        start = timeit.timeit()
        gridmask = np.zeros(len(self.dst1))
        for idx, i in enumerate(self.dst1):
            if i[0] < im_dst.shape[1] and i[0] >0 and i[1] < im_dst.shape[1] and i[1] >refcoor.points[0][1]:
                gridmask[idx] = 1
    
        finalgrid = np.array(self.dst1)[gridmask == 1]
        rowsmask = np.array(self.rowsmask)[gridmask == 1]
        uniquerows = np.unique(rowsmask)
        rowlines = self.rowlines[uniquerows]
        
    #    for i in finalgrid:
    #        cv2.circle(im_dst,(int(i[0]),int(i[1])),3,(255,0,0),-1)
    #    cv2.imshow("image", im_dst)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    
        dropend1a= cv2.convertPointsToHomogeneous(np.array(objRefpts[0]).reshape(-1,1,2)) 
        dropend1b= cv2.convertPointsToHomogeneous(self.zVanish.reshape(-1,1,2)) 
        dropline1 = np.cross(dropend1a, dropend1b)
        
        dropend2a= cv2.convertPointsToHomogeneous(np.array(objRefpts[1]).reshape(-1,1,2)) 
        dropend2b= cv2.convertPointsToHomogeneous(self.zVanish.reshape(-1,1,2)) 
        dropline2 = np.cross(dropend2a, dropend2b)
        
        intersectionA= np.cross(rowlines, dropline1)
        intersectionA= cv2.convertPointsFromHomogeneous(intersectionA)
        intersectionA = intersectionA.reshape([intersectionA.shape[0], intersectionA.shape[2]])
        
        intersectionB= np.cross(rowlines, dropline2)
        intersectionB= cv2.convertPointsFromHomogeneous(intersectionB)
        intersectionB = intersectionB.reshape([intersectionB.shape[0], intersectionB.shape[2]])

        accept = []
        for i in range(len(uniquerows)):
            closeA = np.argmin([abs(j[0] - intersectionA[i][0]) for j in finalgrid[rowsmask==uniquerows[i]]])
            closeB = np.argmin([abs(j[0] - intersectionB[i][0]) for j in finalgrid[rowsmask==uniquerows[i]]])
            if closeB-closeA == self.householdshelterlength:
                accept += [[i, closeA, closeB]]
        
        for idx, i in enumerate(accept):
            if idx == int(len(accept)/2):
                a= finalgrid[[rowsmask==uniquerows[i[0]]]] [[i[1], i[2]]]
                for pt in a:
                    cv2.circle(im_dst,tuple((int(pt[0]), int(pt[1]))),6,(0,255,255))
    
        while(1):
            cv2.imshow('image', im_dst)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        
        end = timeit.timeit()
        diff = end-start
        print("time take to process:")
        print(diff)

if __name__ == '__main__' :
    
    im_dst = cv2.imread("data\\blkcamera.jpg")
    
    zVanish = calibrateVP()

    plane1 = calibratePlane(zVanish)
    plane1.calibrate()
    
    refcoor = CoordinateStore(im_dst)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', refcoor.select_point)
    while(1):
        if len(refcoor.points) != 0:
            for i in refcoor.points:
                cv2.line(im_dst,pt1=tuple(zVanish),pt2=i,color=(0,255,255),thickness=2)

        cv2.imshow('image', im_dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    
    plane1.detectFallArea(refcoor.points, im_dst)


