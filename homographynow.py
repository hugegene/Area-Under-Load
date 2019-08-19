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
        self.rectpts = []

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
    return zVanish

def calibratePlane(zVanish):
    im_dst = cv2.imread("data\\blkcamera.jpg")
    #draw building plane   
    im_src = cv2.imread("data\\blkplan2.jpg")

    polycoor = CoordinateStore(im_src)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', polycoor.draw_rect)
    while(1):
        if len(polycoor.points) >2:
            im_src = cv2.imread("data\\blkplan2.jpg")
#            cv2.polylines(im_src, [np.array(polycoor.points)], True, (0,255,0), thickness=3)
#            cv2.rectangle(im_src, polycoor.rectpts[0], polycoor.rectpts[1], (0,255,0), thickness=3)
#            cv2.line(im_dst,pt1=polycoor.points[0],pt2=polycoor.points[1],color=(0,255,255),thickness=2)
#        cv2.putText(im_src,"Select border of horizontal plane", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imshow('image', im_src)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    
    # Create a source image, a window and bind the function to window    
    rows = np.arange(0, im_src.shape[0], 10)
    cols = np.arange(0, im_src.shape[1], 10)
    points = [[i, j] for j in rows for i in cols]
    print(len(points))

    housholdshelter = [[1146, 395], [1203, 395]]
    householdshelterlength = np.argmin(abs(cols -housholdshelter[1][0]))-np.argmin(abs(cols -housholdshelter[0][0]))
    
#    print(polycoor.rectpts)
    bound = [polycoor.rectpts[0], [polycoor.rectpts[0][0], polycoor.rectpts[1][1]], 
    polycoor.rectpts[1], [polycoor.rectpts[1][0], polycoor.rectpts[0][1]]]
    print(bound)
#    im_src = cv2.imread("data\\blkplan2.jpg")
#    cv2.polylines(im_src, [np.array(bound)], True, (0,255,0), thickness=3)
#    cv2.imshow("Warped bound Image", im_src)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
 
    path = mpltPath.Path(np.array(bound))
    inside2 = path.contains_points(points)
    insidepts = [points[i] for i in range(len(points)) if inside2[i] == True]
    
    uniquecols = np.unique([i[0] for i in insidepts])
    uniquerows = np.unique([i[1] for i in insidepts])

    hello = np.array(insidepts).reshape([len(uniquerows), len(uniquecols), 2])
    print(hello)

#    rowsmask =[]
#    previous = -1
#    a= -1
#    for i in insidepts:
#        if i[1] != previous:
#            a +=1
#            previous = i[1]
#        rowsmask.append(a)
#
#    
#    uniquecols = np.unique([i[0] for i in insidepts])
#    len(uniquecols)
#    
#    colsmask =[[]] * len(uniquecols)
#    for i in insidepts:
#        for j in uniquecols:
#            if j == i[0]:
#                colsmask[]
#        if i[0] == 
#        if i[0] != previous:
#            a +=1
#            previous = i[0]
#        colsmask.append(a)
#    print(colsmask)
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
    print(insidepts)
    print(dst1)
    
    for i in dst1:
        cv2.circle(im_dst,(int(i[0][0]),int(i[0][1])),3,(255,0,0),-1)
    cv2.imshow("image", im_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return householdshelterlength, dst1, [len(uniquerows), len(uniquecols), 2]



if __name__ == '__main__' :
    
    zVanish = calibrateVP()
    
    householdshelterlength, dst1, shape = calibratePlane(zVanish)
    dst1 = dst1.reshape([dst1.shape[0], dst1.shape[2]])
    dst1grid = dst1.reshape(shape)

    im_dst = cv2.imread("data\\blkcamera.jpg")
    
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

    
    average = np.int32(np.average(refcoor.points, 0))
    drop = [average[0], average[1]]
    cv2.circle(im_dst,tuple(average),7,(255,0,0),-1)
    
    minx = abs(dst1grid[:, :, 0] - drop[0])
    minx= np.unravel_index(minx.argmin(), minx.shape)
    
    
    miny = abs(dst1grid[:, :, 1] - drop[1])
    miny = np.unravel_index(miny.argmin(), miny.shape)
    
#    cv2.circle(im_dst,(minx[0],miny[1]),7,(255,0,0),-1)

    n=3
    i,j = minx[0],miny[1]
#    coor2 =  [int(dst1grid[i][j][0]), int(dst1grid[i][j][1])]
#    coor1 =  [int(dst1grid[i][j-n][0]), int(dst1grid[i][j-n][1])]
#    coor3 =  dst1grid[i+n][j]
#    coor4 =  dst1grid[i+n][j-n]

    coor = np.int32(np.array([dst1grid[i][j-n], dst1grid[i][j], dst1grid[i+n][j], dst1grid[i+n][j-n]]))
    
    cv2.polylines(im_dst, [coor], True, (0,255,0), thickness=3)
    cv2.imshow("image", im_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#    gridmask = np.zeros(len(dst1))
#    for idx, i in enumerate(dst1):
#        if i[0][0] < im_dst.shape[1] and i[0][0] >0 and i[0][1] < im_dst.shape[1] and i[0][1] >refcoor.points[0][1]:
#            gridmask[idx] = 1

#    finalgrid = np.array(dst1)[gridmask == 1]
#    finalgrid = finalgrid.reshape([finalgrid.shape[0],finalgrid.shape[2]])
#    rowsmask = np.array(rowsmask)[gridmask == 1]
#    print(len(rowsmask))
   
   
    rowsmask = np.array(rowsmask)
    uniquerows = np.unique(rowsmask)
    rowsgrid = []
    for i in uniquerows[:]:
#        print(dst1[rowsmask == i])
        print([i[0] for i in dst1[rowsmask == i]])
        rowsgrid += [[i[0] for i in dst1[rowsmask == i]]]
        
    
    
    
    
    
    uniquerows = np.unique(rowsmask)
    rowend1= np.array([finalgrid[rowsmask==i][0] for i in uniquerows]).reshape(-1,1,2) 
    rowend2= np.array([finalgrid[rowsmask==i][-1] for i in uniquerows]).reshape(-1,1,2)
    rowend1 = cv2.convertPointsToHomogeneous(rowend1) 
    rowend2 = cv2.convertPointsToHomogeneous(rowend2) 
    rowlines = np.cross(rowend1, rowend2)
    print(len(rowlines))
    
    start = timeit.timeit()
    dropend1a= cv2.convertPointsToHomogeneous(np.array(refcoor.points[0]).reshape(-1,1,2)) 
    dropend1b= cv2.convertPointsToHomogeneous(zVanish.reshape(-1,1,2)) 
    dropline1 = np.cross(dropend1a, dropend1b)
    
    
    dropend2a= cv2.convertPointsToHomogeneous(np.array(refcoor.points[1]).reshape(-1,1,2)) 
    dropend2b= cv2.convertPointsToHomogeneous(zVanish.reshape(-1,1,2)) 
    dropline2 = np.cross(dropend2a, dropend2b)
    
    intersectionA= np.cross(rowlines, dropline1)
    intersectionA= cv2.convertPointsFromHomogeneous(intersectionA)
    intersectionA = intersectionA.reshape([intersectionA.shape[0], intersectionA.shape[2]])
    
    intersectionB= np.cross(rowlines, dropline2)
    intersectionB= cv2.convertPointsFromHomogeneous(intersectionB)
    intersectionB = intersectionB.reshape([intersectionB.shape[0], intersectionB.shape[2]])
    
    
    
    
    

#p=np.array([[2,3],[4,5]],np.float32).reshape(-1,1,2) 
#p2=np.array([[7,6],[8,9]],np.float32).reshape(-1,1,2)
#h1 = cv2.convertPointsToHomogeneous(p) 
#h2 = cv2.convertPointsToHomogeneous(p2)
#np.cross(h1,h2)


    
#    # Display images
#    for pt in finalgrid:
##        print(pt)
#        cv2.circle(im_dst,tuple(pt),3,(0,0,255))
#        
#    for pt in intersectionA:
##        print(tuple((int(pt[0]), int(pt[1]))))
#        cv2.circle(im_dst,tuple((int(pt[0]), int(pt[1]))),3,(0,255,255))
#        
#    for pt in intersectionB:
##        print(tuple((int(pt[0]), int(pt[1]))))
#        cv2.circle(im_dst,tuple((int(pt[0]), int(pt[1]))),3,(0,255,255))
     
#    while(1):
#        cv2.imshow('image', im_dst)
#        k = cv2.waitKey(1) & 0xFF
#        if k == 27:
#            break
#    cv2.destroyAllWindows()

    
    accept = []
    for i in range(len(uniquerows)):
        closeA = np.argmin([abs(j[0] - intersectionA[i][0]) for j in finalgrid[rowsmask==uniquerows[i]]])
        closeB = np.argmin([abs(j[0] - intersectionB[i][0]) for j in finalgrid[rowsmask==uniquerows[i]]])
        print(closeB-closeA)
        if closeB-closeA == householdshelterlength:
            accept += [[i, closeA, closeB]]
            
    for i in accept:
        a= finalgrid[[rowsmask==uniquerows[i[0]]]][i[1]:i[2]]
        print(a)
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
    print(start)
    print(end)
    print(diff)


# regular polygon for testing
#lenpoly = 100
#polygon = [[np.sin(x)+0.5,np.cos(x)+0.5] for x in np.linspace(0,2*np.pi,lenpoly)[:-1]]
#print(polygon)
## random points set of points to test 
#N = 10000
#points = zip(np.random.random(N),np.random.random(N))
#print(points)
#print(len(points))
#path = mpltPath.Path(polygon)
#inside2 = path.contains_points(points)
#print(len(inside2))
#insidepts = [points[i] for i in range(len(points)) if inside2[i] == True]
#print(insidepts)
#    
    
    
    
    
#    print(finalgrid)
#    for i in refcoor:
#        print(refcoor)
#    print(refcoor.points)
#    
#    refcoor.points[0]
#    
#    limitgrid(dst1, , yend, xstart=0, ystart =0):

#    cv2.imshow("grid", im_dst)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

#    plt.imshow(im_dst)
#    plt.scatter([i[0] for i in finalgrid], [i[1] for i in finalgrid], color = 'blue')
#    plt.show()
    
