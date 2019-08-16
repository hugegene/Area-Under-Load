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



def limitgrid(pts, xend, yend, xstart=0, ystart =0):
    out = []
    for i in pts:
#        print(i[0])
        if i[0][0] < xend and i[0][0] >xstart and i[0][1] < yend and i[0][1] >ystart:
#            print("add")
            finalgrid +=[[i[0][0],i[0][1]]]   
    return out

if __name__ == '__main__' :
    
    #instantiate class
    im_dst = cv2.imread("data\\blkcamera.jpg")
    zVanish, xVanish, yVanish = calibrateframe(im_dst)
    zVanish = np.array((int(zVanish[0]), int(zVanish[1])))
    print(zVanish)
    
    
#    cv2.circle(im_dst,(int(zVanish[0]), int(zVanish[1])),3,(255,0,0),-1)
#    cv2.imshow("image", im_dst)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    #draw building plane   
    im_src = cv2.imread("data\\blkplan2.jpg")
#    im_src_copy = im_src.copy()
    polycoor = CoordinateStore(im_src)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', polycoor.select_point)
    while(1):
        
        if len(polycoor.points) >2:
            im_src = cv2.imread("data\\blkplan2.jpg")
            cv2.polylines(im_src, [np.array(polycoor.points)], True, (0,255,0), thickness=3)
#            cv2.line(im_dst,pt1=polycoor.points[0],pt2=polycoor.points[1],color=(0,255,255),thickness=2)

        cv2.imshow('image', im_src)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    

    # Create a source image, a window and bind the function to window    
    rows = np.arange(0, im_src.shape[0], 20)
    cols = np.arange(0, im_src.shape[1], 20)
    points = [[i, j] for j in rows for i in cols]
    print(len(points))


    
    path = mpltPath.Path(polycoor.points)
   
    inside2 = path.contains_points(points)
    insidepts = [points[i] for i in range(len(points)) if inside2[i] == True]
    print(len(insidepts))
    print(insidepts)
    
    
    rowsmask =[]
    previous = 0
    a= 0
    for i in insidepts:
        if i[1] != previous:
            a +=1
            previous = i[1]
        rowsmask.append(a)
    print(len(rowsmask))     
    
    plt.imshow(im_src)
    plt.scatter([i[0] for i in insidepts], [i[1] for i in insidepts], color = 'blue')
    plt.show()
    
    print(insidepts)
    insidepts[0]
    
    coordinateStore1 = CoordinateStore(im_src)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',coordinateStore1.select_point)
    
    
    while(1):
        cv2.imshow('image',im_src)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    
#    print("Selected Source Coordinates: ")
#    for i in coordinateStore1.points:
#        print(i)
        
    # Create a destination image, a window and bind the function to window
    im_dst = cv2.imread("data\\blkcamera.jpg")
    coordinateStore2 = CoordinateStore(im_dst)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',coordinateStore2.select_point)
    
    while(1):
        cv2.imshow('image', im_dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    
    
#    print("Selected Destination Coordinates 2: ")
#    for i in coordinateStore2.points:
#        print(i)
    
#    print(coordinateStore1.points)
#    print(coordinateStore2.points)
    # Calculate Homography
    h, status = cv2.findHomography(np.array(coordinateStore1.points), np.array(coordinateStore2.points))
#    print(h)
#    print(status)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
     
    # Display images
#    cv2.imshow("Source Image", im_src)
#    cv2.imshow("Destination Image", im_dst)
#    cv2.imshow("Warped Source Image", im_out)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    
    pts = np.array(insidepts, np.float32)
    pts1 = pts.reshape(-1,1,2).astype(np.float32)
    dst1 = cv2.perspectiveTransform(pts1, h)


#   finalgrid =[]
#    for i in dst1:
##        print(i[0])
#        if i[0][0] < im_dst.shape[1] and i[0][0] >0 and i[0][1] < im_dst.shape[1] and i[0][1] >0:
##            print("add")
#            finalgrid +=[[i[0][0],i[0][1]]]
##    print(finalgrid)
#    
#    # Display images
#    for pt in finalgrid:
#        cv2.circle(im_dst,tuple(pt),3,(0,0,255))
     
    
    refcoor = CoordinateStore(im_dst)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', refcoor.select_point)

    while(1):
        
        if len(refcoor.points) != 0:
            for i in refcoor.points:
#                print(i)
                try:
                    cv2.line(im_dst,pt1=zVanish,pt2=i,color=(0,255,255),thickness=2)
                except:
                    continue

        cv2.imshow('image', im_dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    
    
    gridmask = np.zeros(len(dst1))
    for idx, i in enumerate(dst1):
#        print(i)
        if i[0][0] < im_dst.shape[1] and i[0][0] >0 and i[0][1] < im_dst.shape[1] and i[0][1] >refcoor.points[0][1]:
#            print("add")
#            finalgrid +=[[i[0][0],i[0][1]]]
            gridmask[idx] = 1
    print(gridmask)
    
  
    finalgrid = np.array(dst1)[gridmask == 1]
    finalgrid = finalgrid.reshape([finalgrid.shape[0],finalgrid.shape[2]])
    rowsmask = np.array(rowsmask)[gridmask == 1]
    print(len(rowsmask))
    print(len(finalgrid))
    
    
    rowend1= np.array([finalgrid[rowsmask==i][0] for i in np.unique(rowsmask)]).reshape(-1,1,2) 
    rowend2= np.array([finalgrid[rowsmask==i][-1] for i in np.unique(rowsmask)]).reshape(-1,1,2)
    rowend1 = cv2.convertPointsToHomogeneous(rowend1) 
    rowend2 = cv2.convertPointsToHomogeneous(rowend2) 
    rowlines = np.cross(rowend1, rowend2)
    rowlines = cv2.convertPointsFromHomogeneous(rowlines)
    
    
    dropend1a= cv2.convertPointsToHomogeneous(np.array(refcoor.points[0]).reshape(-1,1,2)) 
    dropend1b= cv2.convertPointsToHomogeneous(zVanish.reshape(-1,1,2)) 
    dropline1 = np.cross(dropend1a, dropend1b)
    
    
    dropend2a= cv2.convertPointsToHomogeneous(np.array(refcoor.points[1]).reshape(-1,1,2)) 
    dropend2b= cv2.convertPointsToHomogeneous(zVanish.reshape(-1,1,2)) 
    dropline2 = np.cross(dropend2a, dropend2b)
    
    intersectionA= np.cross(rowlines, dropline1)
    intersectionA= cv2.convertPointsFromHomogeneous(intersectionA)
    intersectionA.reshape([intersectionA.shape[0], intersectionA.shape[2]])
    
    intersectionB= np.cross(rowlines, dropline2)
    intersectionB= cv2.convertPointsFromHomogeneous(intersectionB)
    intersectionB.reshape([intersectionB.shape[0], intersectionB.shape[2]])
    
    
    
    
    

#p=np.array([[2,3],[4,5]],np.float32).reshape(-1,1,2) 
#p2=np.array([[7,6],[8,9]],np.float32).reshape(-1,1,2)
#h1 = cv2.convertPointsToHomogeneous(p) 
#h2 = cv2.convertPointsToHomogeneous(p2)
#np.cross(h1,h2)


    
    # Display images
    for pt in finalgrid:
        print(pt)
        cv2.circle(im_dst,tuple(pt),3,(0,0,255))
        
    for pt in intersectionA:
        print(tuple((int(pt[0][0]), int(pt[0][1]))))
        cv2.circle(im_dst,tuple((int(pt[0][0]), int(pt[0][1]))),3,(0,255,255))
        
#    for pt in intersectionB:
#        print(tuple((int(pt[0][0]), int(pt[0][1]))))
#        cv2.circle(im_dst,tuple((int(pt[0][0]), int(pt[0][1]))),3,(0,255,255))
     
    while(1):
        cv2.imshow('image', im_dst)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

print(finalgrid)

print(len(finalgrid))

    


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
    
