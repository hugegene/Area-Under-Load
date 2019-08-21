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
        self.planegrid = []
        

    def select_point(self,event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                print(x,y)
                cv2.circle(self.img,(x,y),3,(255,0,0),-1)
                self.points.append([x,y])
                
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
                
    def draw_rect(self,event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print('Start Mouse Position: '+str(x)+', '+str(y))
                self.rectpts = []
                self.rectpts.append([x,y])
                print(self.rectpts)
                
#                cv2.line(self.img, pt1=(self.line[0]),pt2=(x,y),color=(255,255,0),thickness=2)
                
            elif event == cv2.EVENT_LBUTTONUP:
                print('End Mouse Position: '+str(x)+', '+str(y))
                self.rectpts.append([x,y])
                cv2.rectangle(self.img, tuple(self.rectpts[0]), (x,y), (0,255,0), thickness=3)
                print("rectangle created")
                print(self.rectpts)
                
def calibrateVP():
    im_dst = cv2.imread("data\\blkcamera.jpg")
    zVanish, xVanish, yVanish = calibrateframe(im_dst)
    zVanish = np.array((int(zVanish[0]), int(zVanish[1])))
    xVanish = np.int32(xVanish)
    yVanish = np.int32(yVanish)
    return zVanish, xVanish, yVanish

class calibratePlane():
    def __init__(self, zVanish):
        self.rowlines = []
        self.collines = []
        self.rowsmask = []
        self.householdshelterlength =[]
        self.dst1 = []
        self.xVanish = []
        self.yVanish = []
        self.zVanish =np.squeeze(cv2.convertPointsToHomogeneous(zVanish.reshape(-1,1,2)))

    def calibrate(self, gridsize = 3, dstimage = "data\\blkcamera.jpg", srcimage = "data\\blkplan2.jpg"):    
        im_dst = cv2.imread(dstimage)
        #draw building plane   
        im_src = cv2.imread(srcimage)
    
        polycoor = CoordinateStore(im_src)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', polycoor.draw_rect)
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
        rows = np.arange(0, im_src.shape[0], gridsize)
        cols = np.arange(0, im_src.shape[1], gridsize)
        points = [[i, j] for j in rows for i in cols]
#        print(len(points))
    
        housholdshelter = [[1146, 395], [1203, 395]]
        self.householdshelterlength = np.argmin(abs(cols -housholdshelter[1][0]))-np.argmin(abs(cols -housholdshelter[0][0]))
        
        bound = [polycoor.rectpts[0], [polycoor.rectpts[0][0], polycoor.rectpts[1][1]], 
        polycoor.rectpts[1], [polycoor.rectpts[1][0], polycoor.rectpts[0][1]]]
        
        path = mpltPath.Path(np.array(bound))
        inside2 = path.contains_points(points)
        insidepts = [points[i] for i in range(len(points)) if inside2[i] == True]
        
        uniquecols = np.unique([i[0] for i in insidepts])
        uniquerows = np.unique([i[1] for i in insidepts])

#        rowsmask =[]
#        previous = -1
#        a= -1
#        for i in insidepts:
#            if i[1] != previous:
#                previous = i[1]
#                a +=1
#            rowsmask.append(a)
    
    
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
        self.dst1 = cv2.perspectiveTransform(pts1, h)
        self.dst1 = self.dst1.reshape([self.dst1.shape[0],self.dst1.shape[2]])
        self.planegrid = self.dst1.reshape([len(uniquerows), len(uniquecols), 2])
        
        
        for i in self.dst1:
            cv2.circle(im_dst,(int(i[0]),int(i[1])),3,(255,0,0),-1)
        cv2.imshow("image", im_dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        rowend1= np.array([i[0]for i in self.planegrid])
        rowend2= np.array([i[-1]for i in self.planegrid])
#        print(rowend2)
#        print(rowend2.shape)
        rowend1 = cv2.convertPointsToHomogeneous(rowend1) 
        rowend2 = cv2.convertPointsToHomogeneous(rowend2) 
        
        self.rowlines = np.cross(rowend1, rowend2)
        
        print("sort x y vanishing point")
        rowintersection = np.cross(self.rowlines[0], self.rowlines[1])
        rowintersection = cv2.convertPointsFromHomogeneous(rowintersection)
        self.xVanish = rowintersection.reshape(rowintersection.shape[2],)

#        [self.planegrid[:,i][0] for i in range(self.planegrid.shape[1])]
#        print(self.planegrid)
        colend1= np.array([self.planegrid[:,i][0] for i in range(self.planegrid.shape[1])])
#        print("colend1")
#        print(colend1)
        colend2= np.array([self.planegrid[:,i][-1] for i in range(self.planegrid.shape[1])])
#        print("colend2")
#        print(colend2)
#        print(colend2.shape)
        colend1 = cv2.convertPointsToHomogeneous(colend1) 
        colend2 = cv2.convertPointsToHomogeneous(colend2) 
#        print("homographed")
#        print(colend1)
#        print(colend2)
        self.collines = np.cross(colend1, colend2)
        colintersection = np.cross(self.collines[0], self.collines[1])
        colintersection = cv2.convertPointsFromHomogeneous(colintersection)
        self.yVanish = colintersection.reshape(colintersection.shape[2],)


    def detectFallArea(self, objRefpts, im_dst):
        
        start = timeit.timeit()
        print(start)
        
        print("checking--------------")
        print(objRefpts)
        print(np.array(objRefpts))
        print(self.xVanish)
        
      
#        midpts = []
#        edgeVector = []
#        for i in range(len(objRefpts)-1): 
#            midpts += [np.average(objRefpts[i:i+2], axis =0)]
#            edgeVector += [np.array(objRefpts[i])- np.array(objRefpts[i+1])]
#        print(midpts)
        
        midpts =  [np.average(objRefpts[0:2], axis =0)]
        edgeVector = [np.array(objRefpts[0])- np.array(objRefpts[1])]
            
        print("array")

        print(np.array(edgeVector))

#        for midpt in midpts:
#            cv2.circle(im_dst,(int(midpt[0]),int(midpt[1])),3,(255,0,0),-1)
#        cv2.imshow("image", im_dst)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
        print(midpts)
        print(np.array(self.xVanish))
        
        xVVector = midpts - np.array(self.xVanish)
        yVVector = midpts - np.array(self.yVanish)
        
        print(np.transpose(midptVector))
        dot = np.dot(np.array(edgeVector), np.transpose(midptVector))
        print(dot)
        
        
        mag = [abs(np.linalg.norm(edgeVector[i]))*abs(np.linalg.norm(midptVector[i])) for i in range(len(midptVector))]
        print(mag)
        
        compare = dot[np.eye(len(midptVector)) ==1]/np.array(mag)
        
        
        if compare[0]>compare[1]:
            align = ("x", "y")
        else:
            align = ("y", "x")
        
     
        
        
        
        return dot, align, objRefpts, self.zVanish, self.rowlines, self.planegrid, self.householdshelterlength
        
        
#        
##        checkVanish = np.dot(np.array(objRefpts), np.array([xVanish, yVanish]))
##        print(checkVanish)
#        
#        dropend1a= cv2.convertPointsToHomogeneous(np.array(objRefpts[0]).reshape(-1,1,2)) 
##        dropline1 = np.cross(dropend1a, self.zVanish)
#        
#        dropend2a= cv2.convertPointsToHomogeneous(np.array(objRefpts[1]).reshape(-1,1,2)) 
##        dropend3a= cv2.convertPointsToHomogeneous(np.array(objRefpts[2]).reshape(-1,1,2)) 
##        dropline2 = np.cross(dropend2a, self.zVanish)
#        
##        dropends = cv2.convertPointsToHomogeneous(np.array([objRefpts[0], objRefpts[1]]).reshape(-1,1,2))
##        print(dropends)
#        a = np.array([dropend1a.reshape(3,), dropend2a.reshape(3,)])
#        b = np.array([self.zVanish.reshape(3,), self.zVanish.reshape(3,)])
#        droplines = np.cross(a,b)
#        print(droplines)
##        print(np.cross(dropends, b))
#        
##        print("intersections")
#        intersections= np.cross(self.rowlines, droplines)
##        print(intersections)
##        print(intersections.shape)
#        
##        print("interectionsA")
##        intersectionA= np.cross(self.rowlines, dropline1)
##        print(intersectionA)
##        print(intersectionA.shape)
##        intersectionA= cv2.convertPointsFromHomogeneous(intersectionA)
##        intersectionA = intersectionA.reshape([intersectionA.shape[0], intersectionA.shape[2]])
#        
##        print("altInterA")
##        print(intersections[:,0,:])
#        intersectionA= cv2.convertPointsFromHomogeneous(intersections[:,0,:])
#        intersectionA = intersectionA.reshape([intersectionA.shape[0], intersectionA.shape[2]])
##        print(intersections)
#        intersectionB= cv2.convertPointsFromHomogeneous(intersections[:,1,:])
#        intersectionB = intersectionB.reshape([intersectionB.shape[0], intersectionB.shape[2]])
#
##        print("interectionsB")
##        intersectionB= np.cross(self.rowlines, dropline2)
##        print(intersectionB)
##        print(intersectionB.shape)
##        intersectionB= cv2.convertPointsFromHomogeneous(intersectionB)
##        intersectionB = intersectionB.reshape([intersectionB.shape[0], intersectionB.shape[2]])
#
#     
#        accept = []
#
#        print(self.householdshelterlength)
#        for i in range(self.planegrid.shape[0]):
#            closeA = np.argmin([abs(j[0] - intersectionA[i][0]) for j in self.planegrid[i]])
#            closeB = np.argmin([abs(j[0] - intersectionB[i][0]) for j in self.planegrid[i]])
#            if closeB-closeA == self.householdshelterlength:
#                accept += [[i, closeA, closeB]]
#
#        
#        for idx, i in enumerate(accept):
#            if idx == int(len(accept)/2):
#                a= np.int32(self.planegrid[i[0]][[i[1], i[2]]])
#                print(a)
#                cv2.line(im_dst,pt1=tuple(a[0]),pt2=tuple(a[1]),color=(0,255,255),thickness=2)
##                a= finalgrid[[rowsmask==uniquerows[i[0]]]] [[i[1], i[2]]]
#
#        end = timeit.timeit()
#        print(end)
#        diff = end-start
#        print("time take to process:")
#        print(diff)
#        while(1):
#            cv2.imshow('image', im_dst)
#            k = cv2.waitKey(1) & 0xFF
#            if k == 27:
#                break
#        cv2.destroyAllWindows()
        
        

if __name__ == '__main__' :
    
    im_dst = cv2.imread("data\\blkcamera.jpg")
    
    zVanish, xVanish, yVanish = calibrateVP()

    plane1 = calibratePlane(zVanish)
    plane1.calibrate()
    
    refcoor = CoordinateStore(im_dst)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', refcoor.select_point)
    while(1):
        if len(refcoor.points) != 0:
            for i in refcoor.points:
                cv2.line(im_dst,pt1=tuple(zVanish),pt2=tuple(i),color=(0,255,255),thickness=2)

        cv2.imshow('image', im_dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    
    dot, align, objRefpts, zVanish, rowlines, planegrid, householdshelterlength = plane1.detectFallArea(refcoor.points, im_dst)
    dot[np.eye(2) ==1]
    dot[np.eye(2) ==1]/np.array([1,2])
    np.eye(1)
    
    print(align)
    print(zVanish)
    
    
     
        
#        checkVanish = np.dot(np.array(objRefpts), np.array([xVanish, yVanish]))
#        print(checkVanish)
    
    homorefpts = cv2.convertPointsToHomogeneous(np.array(objRefpts))
    homorefpts = np.squeeze(homorefpts)


    dropend1a= cv2.convertPointsToHomogeneous(np.array(objRefpts[0]).reshape(-1,1,2)) 
    dropend2a= cv2.convertPointsToHomogeneous(np.array(objRefpts[1]).reshape(-1,1,2))
    dropend3a= cv2.convertPointsToHomogeneous(np.array(objRefpts[2]).reshape(-1,1,2)) 
    print(dropend3a)
#        dropend3a= cv2.convertPointsToHomogeneous(np.array(objRefpts[2]).reshape(-1,1,2)) 
#        dropline2 = np.cross(dropend2a, self.zVanish)
    
#        dropends = cv2.convertPointsToHomogeneous(np.array([objRefpts[0], objRefpts[1]]).reshape(-1,1,2))
#        print(dropends)
#    a = np.array([dropend1a.reshape(3,), dropend2a.reshape(3,)])
#    b = np.array([self.zVanish.reshape(3,), self.zVanish.reshape(3,)])
#    
    b =np.repeat([zVanish], len(homorefpts), axis =0)

    droplines = np.cross(homorefpts,b)
    print(droplines)
    
#        print("intersections")
    intersections= np.cross(rowlines, droplines)
    intersections.shape
    intersections[:,0,:].shape
#        print(intersections)
#        print(intersections.shape)
    
#        print("interectionsA")
#        intersectionA= np.cross(self.rowlines, dropline1)
#        print(intersectionA)
#        print(intersectionA.shape)
#        intersectionA= cv2.convertPointsFromHomogeneous(intersectionA)
#        intersectionA = intersectionA.reshape([intersectionA.shape[0], intersectionA.shape[2]])
    
#        print("altInterA")
#        print(intersections[:,0,:])
    
    intersect = []
    for i in range(intersections.shape[1]):
        intersectionA= cv2.convertPointsFromHomogeneous(intersections[:,i,:])
        intersect += np.squeeze(intersectionA)

#    intersectionB= cv2.convertPointsFromHomogeneous(intersections[:,1,:])
#    intersectionB = intersectionB.reshape([intersectionB.shape[0], intersectionB.shape[2]])

#        print("interectionsB")
#        intersectionB= np.cross(self.rowlines, dropline2)
#        print(intersectionB)
#        print(intersectionB.shape)
#        intersectionB= cv2.convertPointsFromHomogeneous(intersectionB)
#        intersectionB = intersectionB.reshape([intersectionB.shape[0], intersectionB.shape[2]])

    print(householdshelterlength)
    accept = []

    for i in range(planegrid.shape[0]):
        closeA = np.argmin([abs(j[0] - intersectionA[i][0]) for j in self.planegrid[i]])
        closeB = np.argmin([abs(j[0] - intersectionB[i][0]) for j in self.planegrid[i]])
        if closeB-closeA == self.householdshelterlength:
            accept += [[i, closeA, closeB]]

    
    for idx, i in enumerate(accept):
        if idx == int(len(accept)/2):
            a= np.int32(self.planegrid[i[0]][[i[1], i[2]]])
            print(a)
            cv2.line(im_dst,pt1=tuple(a[0]),pt2=tuple(a[1]),color=(0,255,255),thickness=2)
#                a= finalgrid[[rowsmask==uniquerows[i[0]]]] [[i[1], i[2]]]

    end = timeit.timeit()
    print(end)
    diff = end-start
    print("time take to process:")
    print(diff)
    while(1):
        cv2.imshow('image', im_dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


