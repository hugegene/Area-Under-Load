# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:18:25 2019

@author: bdgecyt
"""

import cv2
import numpy as np
import copy
import math
import Edges
import wrapper


imagename =  "data\\25.jpg"
image = cv2.imread(imagename)
orEdges = Edges.getEdges(image)
print ("got edges")
orLines = Edges.getLines(orEdges, 0)
print ("got lines , num : " + str(len(orLines)))


#Read gray image
#img = cv2.imread("test.png",0)
#
##Create default parametrization LSD
#lsd = cv2.createLineSegmentDetector(0)
#
##Detect lines in the image
##lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
#dlines = lsd.detect(img)
##Draw detected lines in the image
#drawn_img = lsd.drawSegments(img,lines)
#
##Show image
#cv2.imshow("LSD",drawn_img )
#cv2.waitKey(0)


#
def drawLines(image,lines,color = (0,0,255),width = 2):
    for item in lines:
        cv2.line(image,(item[0],item[1]),(item[2],item[3]),color,width)
    return image

image = drawLines(image, orLines)
#
#
while(True):
        cv2.imshow("frame", image)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break

    
    
#exLines = Edges.extenLines(orLines,orEdges)
#print ("extend lines")
#exLines = Edges.mergeLines(exLines)