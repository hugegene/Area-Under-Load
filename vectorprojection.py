# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:08:21 2019

@author: eugene
"""

from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sucestest import calibrateframe
import matplotlib.path as mpltPath
import time

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
        self.zVanish =zVanish

    def calibrate(self, gridsize = 3, dstimage = "data\\blkcamera.jpg", srcimage = "data\\blkplan2.jpg", srcpts =[], dstpts = []):    
        if dstimage == "data\\blkcamera.jpg":
            im_dst = cv2.imread(dstimage)
        else:
            im_dst = dstimage
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
        for pt in points:
            cv2.circle(im_src,(int(pt[0]),int(pt[1])),3,(255,255,0),-1)
        
        cv2.imshow("IIImage", im_src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
            
#        housholdshelter = [[1146, 346] ,[1201, 395]]
        housholdshelter = [[599, 365], [635, 399]]
        self.householdshelterwidth = np.argmin(abs(cols -housholdshelter[1][0]))-np.argmin(abs(cols -housholdshelter[0][0]))
        self.householdshelterlength = np.argmin(abs(rows -housholdshelter[1][1]))-np.argmin(abs(rows -housholdshelter[0][1]))
        
        
        bound = [polycoor.rectpts[0], [polycoor.rectpts[0][0], polycoor.rectpts[1][1]], 
        polycoor.rectpts[1], [polycoor.rectpts[1][0], polycoor.rectpts[0][1]]]
        
        path = mpltPath.Path(np.array(bound))
        inside2 = path.contains_points(points)
        insidepts = [points[i] for i in range(len(points)) if inside2[i] == True]
        
        uniquecols = np.unique([i[0] for i in insidepts])
        uniquerows = np.unique([i[1] for i in insidepts])

        plt.imshow(im_src)
        plt.scatter([i[0] for i in insidepts], [i[1] for i in insidepts], color = 'blue')
        plt.show()
        
        if srcpts == []:
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
        else:
            coordinateStore1 = CoordinateStore(im_src)
            coordinateStore1.points = srcpts
            coordinateStore2 = CoordinateStore(im_src)
            coordinateStore2.points = dstpts
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
        
        print("planegrid")
        print(self.planegrid.shape)
        
        
        for i in self.dst1:
            cv2.circle(im_dst,(int(i[0]),int(i[1])),3,(255,255,0),-1)
            
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
        print("planegrid")
        print(self.planegrid.shape)


    def detectFallArea(self, objRefpts, im_dst):
        
        b =np.int64(np.repeat([self.zVanish], len(objRefpts), axis =0))
        droplines = b-objRefpts
    
        l= self.householdshelterlength
        w= self.householdshelterwidth
        
        pt1 = self.planegrid[1:self.planegrid.shape[0]-l, 1:self.planegrid.shape[1]-w, :]
        
        pt2 = self.planegrid[1+l:self.planegrid.shape[0]-l+l, 1:self.planegrid.shape[1]-w:]
        
        pt3 = self.planegrid[1+l:self.planegrid.shape[0]-l+l,1+w:self.planegrid.shape[1]-w+w:]
        
        pt4 = self.planegrid[1:self.planegrid.shape[0]-l,1+w:self.planegrid.shape[1]-w+w:]
        
        def perpendicular( a ) :
            c = []
            for i in a:
                b = np.empty_like(i)
                b[0] = -i[1]
                b[1] = i[0]
                c+= [b]
            return np.array(c)
        
        dl = perpendicular(droplines)
        
#        np.array([pt1.reshape((-1, pt1.shape[2])), pt2.reshape((-1, pt2.shape[2])), pt3.reshape((-1, pt3.shape[2]))])
 
        x = np.array([pt1.reshape((-1, pt1.shape[2])), pt2.reshape((-1, pt2.shape[2])), pt3.reshape((-1, pt3.shape[2]))]) -objRefpts[0:3].reshape([3,1,2])
    #    pt2
    #    pt2.reshape((-1, pt2.shape[2])).shape
#        x = pt1.reshape((-1, pt1.shape[2]))- objRefpts[0]
#        x2 = pt2.reshape((-1, pt2.shape[2]))- objRefpts[1]
#        x = np.array([x,x2])
#        x.shape
    #    x2 = pt2.reshape((-1, pt2.shape[2]))- objRefpts[1]
        xv = np.dot(x, np.transpose(dl[0:3]))
        vv = np.linalg.norm(dl[0:3], axis =1)

        d = xv/vv.reshape([3,1,1])
#        print(d)
        
        shortestdist1= abs(d[0,:,0]).reshape([pt1.shape[0], pt1.shape[1]])
        shortestdist2= abs(d[1,:,1]).reshape([pt2.shape[0], pt2.shape[1]])
        shortestdist3= abs(d[2,:,2]).reshape([pt3.shape[0], pt3.shape[1]])
        shortestdist =  shortestdist1 +  shortestdist2 + shortestdist3
        shortestdist.shape
        shortestpoint = np.unravel_index(shortestdist.argmin(), shortestdist.shape)

    #    pts = np.int32(plane1.planegrid.reshape([plane1.planegrid.shape[0]* plane1.planegrid.shape[1], -1]))
    #    for i in pts:
    #        cv2.circle(im_dst, tuple(i) ,3,(255,0,0),-1)
       
    #    cv2.circle(im_dst, tuple(np.int32(pt4[0][0])) ,8,(255,255,0),-1)
    
        cv2.circle(im_dst, tuple(np.int32(pt1[shortestpoint[0]][shortestpoint[1]])) ,8,(255,0,0),-1)
        
        cv2.circle(im_dst, tuple(np.int32(pt2[shortestpoint[0]][shortestpoint[1]])) ,8,(255,0,0),-1)
        
        cv2.circle(im_dst, tuple(np.int32(pt3[shortestpoint[0]][shortestpoint[1]])) ,8,(255,0,0),-1)
        
        cv2.circle(im_dst, tuple(np.int32(pt4[shortestpoint[0]][shortestpoint[1]])) ,8,(255,0,0),-1)
        
    #    pts= np.array([pt2[1][1], pt1[1][1],  pt3[1][1], pt4[1][1]], np.int32)
    #    cv2.polylines(im_dst, [pts], True, (0,255,0), thickness=3)
        
        area = np.array([tuple(np.int32(pt1[shortestpoint[0]][shortestpoint[1]])), 
         tuple(np.int32(pt2[shortestpoint[0]][shortestpoint[1]])),
         tuple(np.int32(pt3[shortestpoint[0]][shortestpoint[1]])),
         tuple(np.int32(pt4[shortestpoint[0]][shortestpoint[1]]))])
            
#        cv2.imshow("image", im_dst)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
        return area

if __name__ == '__main__' :
    
    im_dst = cv2.imread("data\\blkcamera.jpg")
    
    zVanish, xVanish, yVanish = calibrateVP()
#    print(zVanish)
#    print(plane1.zVanish)

    plane1 = calibratePlane(zVanish)
    plane1.calibrate(gridsize = 15, dstimage = im_dst)
    
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
#    objRefpts = np.array(refcoor.points, np.int64)
    plane1.detectFallArea(np.array(refcoor.points, np.int64), im_dst)


#    plane1.householdshelterlength
#    plane1.householdshelterwidth
    
#    objRefpts.dtype
#    plane1.planegrid.shape
#
#    b =np.int64(np.repeat([zVanish], len(objRefpts), axis =0))
#    droplines = b-objRefpts
#
#    l= plane1.householdshelterlength
#    w= plane1.householdshelterwidth
#    
#    pt1 = plane1.planegrid[1:plane1.planegrid.shape[0]-l, 1:plane1.planegrid.shape[1]-w, :]
#    
#    pt2 = plane1.planegrid[1+l:plane1.planegrid.shape[0]-l+l, 1:plane1.planegrid.shape[1]-w:]
#    
#    pt3 = plane1.planegrid[1+l:plane1.planegrid.shape[0]-l+l,1+w:plane1.planegrid.shape[1]-w+w:]
#    
#    pt4 = plane1.planegrid[1:plane1.planegrid.shape[0]-l,1+w:plane1.planegrid.shape[1]-w+w:]
#    
#    def perpendicular( a ) :
#        c = []
#        for i in a:
#            b = np.empty_like(i)
#            b[0] = -i[1]
#            b[1] = i[0]
#            c+= [b]
#        return np.array(c)
#    
#    dl = perpendicular(droplines)
#    
#    
#    np.array([pt1.reshape((-1, pt1.shape[2])), pt2.reshape((-1, pt2.shape[2]))]).shape
#    
#    objRefpts[0:2].reshape([2,1,2])
#    
#    x=np.array([pt1.reshape((-1, pt1.shape[2])), pt2.reshape((-1, pt2.shape[2]))])-objRefpts[0:2].reshape([2,1,2])
##    pt2
##    pt2.reshape((-1, pt2.shape[2])).shape
#    x = pt1.reshape((-1, pt1.shape[2]))- objRefpts[0]
#    x2 = pt2.reshape((-1, pt2.shape[2]))- objRefpts[1]
#    x = np.array([x,x2])
#    x.shape
##    x2 = pt2.reshape((-1, pt2.shape[2]))- objRefpts[1]
#    xv = np.dot(x, np.transpose(dl[0:2]))
#    xv.shape
#    
#    vv = np.linalg.norm(dl[0:2], axis =1)
##    xv = np.dot(homopt2, np.transpose(droplines[0]))
##    vv = np.dot(dl, np.transpose(dl))
#    xv.shape
#    vv.shape
#    
#    d = xv/vv.reshape([2,1,1])
#    d.dtype
#    d.shape
#    abs(d)
#    d.shape
#    
#    shortestdist1= abs(d[0,:,0]).reshape([pt1.shape[0], pt1.shape[1]])
#    shortestdist2= abs(d[1,:,1]).reshape([pt2.shape[0], pt2.shape[1]])
#    shortestdist =  shortestdist1+  shortestdist2
#    shortestdist.shape
#    shortestpoint = np.unravel_index(shortestdist.argmin(), shortestdist.shape)
#    
#    
##    pts = np.int32(plane1.planegrid.reshape([plane1.planegrid.shape[0]* plane1.planegrid.shape[1], -1]))
##    for i in pts:
##        cv2.circle(im_dst, tuple(i) ,3,(255,0,0),-1)
#   
##    cv2.circle(im_dst, tuple(np.int32(pt4[0][0])) ,8,(255,255,0),-1)
#    cv2.circle(im_dst, tuple(np.int32(pt1[shortestpoint[0]][shortestpoint[1]])) ,8,(255,0,0),-1)
#    
#    cv2.circle(im_dst, tuple(np.int32(pt2[shortestpoint[0]][shortestpoint[1]])) ,8,(255,0,0),-1)
#    
#    cv2.circle(im_dst, tuple(np.int32(pt3[shortestpoint[0]][shortestpoint[1]])) ,8,(255,0,0),-1)
#    
#    cv2.circle(im_dst, tuple(np.int32(pt4[shortestpoint[0]][shortestpoint[1]])) ,8,(255,0,0),-1)
#    
##    pts= np.array([pt2[1][1], pt1[1][1],  pt3[1][1], pt4[1][1]], np.int32)
##    cv2.polylines(im_dst, [pts], True, (0,255,0), thickness=3)
#    
#    cv2.imshow("image", im_dst)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
