# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:46:58 2019

@author: bdgecyt
"""

import cv2
import math
from time import time
import numpy as np
import wrapper
from operator import itemgetter


boxes = []

xCount = 0
yCount = 0
iter = 0
img = 0

def on_mouse(event, x, y, flags, params):
    
    global iter
    t = time()
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: '+str(x)+', '+str(y))
        sbox = [x, y]
        boxes.append(sbox)
        
#        cv2.line(img,pt1=(0,0),pt2=(x,y),color=(255,255,0),thickness=2)
        

    elif event == cv2.EVENT_LBUTTONUP:
        print('End Mouse Position: '+str(x)+', '+str(y))
        ebox = [x, y]
        boxes.append(ebox)
        # print boxes
        iter += 1
        # print iter

def split(start, end, segments):
    x_delta = (end[0] - start[0]) / float(segments)
    y_delta = (end[1] - start[1]) / float(segments)
    points = []
    for i in range(1, segments):
        points.append([start[0] + i * x_delta, start[1] + i * y_delta])
    return [start] + points + [end]
   

def line_intersection(line1, line2):

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) 

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def norm(point1, point2):

    xdiff = point1[0] - point2[0]
    ydiff = point1[1] - point2[1]

    norm = math.sqrt(xdiff*xdiff + ydiff*ydiff)
    # print norm
    return norm


def orderptinline(pts, vp):
#    print("ordering points")
#    print(pts)
    lengths = [norm(pt, vp) for pt in pts]
    lengths= np.argsort(lengths)[::-1]
    strlength = ''.join(str(e) for e in lengths)
#    print(strlength)
    return strlength
        

def getborderpt(line1, line2):
    return line_intersection(line1, line2)

def findAnglebetVP(line, vp):
    a = np.array(line[0])
    b = np.array(line[1])
    c = np.array(vp)
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def estimatelength(order,a1,a2,r1,r2, Vanish, response):
    if order == "0123":
#        print("order is:" + order)
        reflength = (norm(a1, r2)/norm(a1, Vanish))/(norm(r1,r2)/norm(r1,Vanish))*response
#        print(reflength)
        ref2length = (norm(a2, r2)/norm(a2, Vanish))/(norm(r1,r2)/norm(r1,Vanish))*response
#        print(ref2length)
        finallength = reflength-ref2length
    elif order == "0213":
#        print("order is:" + order)
        reflength = (norm(a1, r2)/norm(a1, Vanish))/(norm(r1,r2)/norm(r1,Vanish))*response
        ref2length = response/(norm(r1, r2)/norm(r1, Vanish))/(norm(a2,r2)/norm(a2,Vanish))
        finallength = reflength - ref2length
    elif order == "0213":
        reflength = (norm(a1, r2)/norm(a1, Vanish))/(norm(r1,r2)/norm(r1,Vanish))*response
        ref2length = response/((norm(r1, a2)/norm(r1, Vanish))/(norm(r2,a2)/norm(r2,Vanish))-1)
        finallength = reflength + ref2length
    elif order == "2031":
        reflength = response/(norm(r1, r2)/norm(r1, Vanish))/(norm(a1,r2)/norm(a1,Vanish))
        ref2length = reflength/((norm(a1, a2)/norm(a1, Vanish))/(norm(r2,a2)/norm(r2,Vanish))-1)
        finallength = reflength + ref2length
    elif order == "2301":
        reflength = response/((norm(r1, a1)/norm(r1, Vanish))/(norm(r2,a1)/norm(r2,Vanish))-1)
        ref2length = (reflength +response)/((norm(r1, a2)/norm(r1, Vanish))/(norm(a1,a2)/norm(a2,Vanish))-1)
        finallength = ref2length
    else:
        finallength = 99999
    return finallength 


def calibrateframe(img, findref = False):
    vps = wrapper.dealAImage(img,"data/result/",True,True,True) 
    vps = [[i[0], i[1]] for i in vps]
    print(vps)
    count = 0
#    while(True):
#        
#        # print count
#        if iter == 2:
#            cv2.destroyAllWindows()
#            break
#            
#        count += 1
#        cv2.namedWindow('real image')
#        cv2.setMouseCallback('real image', on_mouse, 0)
#        
#        if len(boxes) != 0:
#            for i in range(0,len(boxes), 2):
#    #            print(i)
#                try:
#                    cv2.line(img,pt1=tuple(boxes[i]),pt2=tuple(boxes[i+1]),color=(0,255,255),thickness=2)
#                   
#                except:
#                    continue
#        cv2.imshow('real image', img)
#        if cv2.waitKey(1) == 27:
#            cv2.destroyAllWindows()
#            break
    print(vps)
    vps = sorted(vps, key=itemgetter(1)) 
    print(vps)
    print(boxes)
    xVanish = vps[0]
    print ("x vanishing pt:" + str(xVanish))
    
    yVanish =  vps[1]
    print ("y vanishing pt:" + str(yVanish))
    
    zVanish =  vps[2]
    print ("z vanishing pt:" + str(zVanish))
    
    if findref == True:
        referenceline = [boxes[0], boxes[1]]
        referenceline.sort(key = lambda x: norm(x, xVanish), reverse = False)
        
        ang1 = findAnglebetVP(referenceline, xVanish)
        print("angles between reference line and xVanish:" + str(ang1))
        referenceline.sort(key = lambda x: norm(x, yVanish), reverse = False)
        ang2 = findAnglebetVP(referenceline, yVanish)
        print("angles between reference line and yVanish:" + str(ang2))
        if ang1> ang2:
            print("ref vp is Y vanishing point" )
            refV= yVanish
            ortV= xVanish
        if ang2> ang1:
            print("ref vp is X vanishing point" )
            refV= xVanish
            ortV= yVanish
            
        referenceline.sort(key = lambda x: norm(x, refV), reverse = True)
        
        estimateline = [boxes[2], boxes[3]]
        estimateline.sort(key = lambda x: norm(x, refV), reverse = True)
        
        response = float(input("Please enter length of reference object: "))
        response2 = float(input("Please enter length of measured object: "))
        
        return response, response2, estimateline, referenceline, refV, ortV, zVanish, xVanish, yVanish
    else:
        return zVanish, xVanish, yVanish

def drawfallarea(img, refV, ortV, zVanish, correctpt, correct2pt):
    nextpt= [int(0.78*img_shape[1]), 
                 int(0.615*img_shape[0])]
    
    
    droptoVP3 = [nextpt, zVanish]
    print("vp3")
    print(droptoVP3)
    bordervp3= line_intersection(droptoVP3, [(0, img_shape[0]),(img_shape[1], img_shape[0])])
    dropline3 = [nextpt, bordervp3]
    
    ptB = line_intersection(dropline3, [correctpt, ortV])
    cv2.line(img,(int(correctpt[0]), int(correctpt[1])), (int(ptB[0]), int(ptB[1])),color=(0,0,255),thickness=2)
    
    backline1 = [correct2pt, ortV]
    backline2 = [ptB, refV]
    ptC= line_intersection(backline1, backline2)
    cv2.line(img,(int(correct2pt[0]), int(correct2pt[1])), (int(ptC[0]), int(ptC[1])),color=(0,0,255),thickness=2)
    cv2.line(img,(int(ptB[0]), int(ptB[1])), (int(ptC[0]), int(ptC[1])),color=(0,0,255),thickness=2)
    
    

def processframe(img, response, response2, estimateline, referenceline, refV, ortV, zVanish, xVanish, yVanish, img_shape):
    
    
    droptoVP1= [estimateline[0], zVanish]
    droptoVP2= [estimateline[1], zVanish]
    print("vp1")
    print(droptoVP1)
    
#    print(droptoVP1)
#    cv2.line(img,(0, int(0.9*img_shape[0])), (img_shape[1], int(0.9*img_shape[0])),color=(0,255,255),thickness=10)
    
    #test line
#    cv2.line(img,(0, int(0.8*img_shape[0])), (int(0.78*img_shape[1]), int(0.615*img_shape[0])),color=(0,255,255),thickness=10)
    
    bordervp1= line_intersection(droptoVP1, [(0, img_shape[0]),(img_shape[1], img_shape[0])])
    bordervp2= line_intersection(droptoVP2, [(0, img_shape[0]),(img_shape[1], img_shape[0])])
    
#    print(bordervp1)
#    print(bordervp2)
    dropline1 = [estimateline[0], bordervp1]
    dropline2 = [estimateline[1], bordervp2]
    
    refline1 = [referenceline[0],ortV]
    refline2 = [referenceline[1],ortV]
    
    print("breaking drop line to segments")
    dropline1seg = split(dropline1[0], dropline1[1], 50)
#    print(dropline1seg)

    finallengths = []
    dropline2pts = []
    for pt in dropline1seg:
    #    print(pt)
        cv2.circle(img,(int(pt[0]), int(pt[1])), 3, (0,255,255), -1)
    #    cv2.line(img,(int(pt[0]), int(pt[1])), (int(yVanish[0]), int(yVanish[1])),color=(0,255,255),thickness=2)
        intersectDropline2= line_intersection([pt, refV], dropline2)
        dropline2pts += [intersectDropline2]
        intersectRefline1= line_intersection([pt, refV], refline1)
        intersectRefline2= line_intersection([pt, refV], refline2)
        cv2.circle(img,(int(intersectDropline2[0]), int(intersectDropline2[1])), 3, (255,0,0), -1)
        cv2.circle(img,(int(intersectRefline1[0]), int(intersectRefline1[1])), 3, (0,255,0), -1)
        cv2.circle(img,(int(intersectRefline2[0]), int(intersectRefline2[1])), 3, (0,0,255), -1)
        
        ordered = orderptinline([pt, intersectDropline2,intersectRefline1, intersectRefline2] , refV)
        finallength = estimatelength(ordered, pt, intersectDropline2,intersectRefline1, intersectRefline2, refV, response)
    #    reflength = (norm(pt, intersectRefline2)/norm(pt, yVanish))/(norm(intersectRefline1,intersectRefline2)/norm(intersectRefline1,yVanish))*response
    #    print(reflength)
    #    ref2length = (norm(intersectDropline2, intersectRefline2)/norm(intersectDropline2, yVanish))/(norm(intersectRefline1,intersectRefline2)/norm(intersectRefline1,yVanish))*response
    #    print(ref2length)
    #    finallength = reflength-ref2length
    #    print("finallength:" +str(finallength))
        finallengths += [finallength]
    
    measurements = [abs(response2- i)for i in finallengths]
    correctpt = dropline1seg[np.argmin(measurements)]
    correct2pt = dropline2pts[np.argmin(measurements)]
    #if finallength  >16 and finallength <18:
    cv2.line(img,(int(estimateline[0][0]), int(estimateline[0][1])), (int(estimateline[1][0]), int(estimateline[1][1])),color=(0,255,255),thickness=2)
    cv2.line(img,(int(correctpt[0]), int(correctpt[1])), (int(correct2pt[0]), int(correct2pt[1])),color=(0,0,255),thickness=2)
    drawfallarea(img, refV, ortV, zVanish, correctpt, correct2pt)
    
    print("nearest measurement:" +str( finallengths[np.argmin(measurements)] ) )
    
    if zVanish:
        cv2.line(img,(int(0.5*img.shape[1]), int(0.5*img.shape[0])), (int(zVanish[0]), int(zVanish[1])),color=(0,255,255),thickness=2)
    if xVanish:
        cv2.line(img,(int(0.5*img.shape[1]), int(0.5*img.shape[0])), (int(xVanish[0]), int(xVanish[1])),color=(0,255,255),thickness=2)
    if yVanish:
        cv2.line(img,(int(0.5*img.shape[1]), int(0.5*img.shape[0])), (int(yVanish[0]), int(yVanish[1])),color=(0,255,255),thickness=2)
    
#    return img

        
if __name__ == "__main__":
    img = cv2.imread('data\\18.jpg')
#    img = cv2.resize(img, None, fx = 0.3,fy = 0.3)
    img_shape = img.shape
#    cv2.circle(img, (100,900), 5, (0,0,255), 5)
#    while(True):
#        cv2.imshow('points image', img)
#        if cv2.waitKey(1) == 27:
#            cv2.destroyAllWindows()
#            break
#    print(img.shape)
    response, response2, estimateline, referenceline, refV, ortV, zVanish, xVanish, yVanish = calibrateframe(img, findref = True)
#    
    while(True):
        print(img.shape)
        img = cv2.imread('data\\18.jpg')
#        img = cv2.resize(img, None, fx = 0.3,fy = 0.3)
        processframe(img, response, response2, estimateline, referenceline, refV, ortV, zVanish, xVanish, yVanish, img_shape)
        cv2.imshow('points image', img)
#        estimateline[0][0] -= 1
#        estimateline[1][0] -= 1
#        print("estimate line is:" + str(estimateline))
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
#     img = cv2.blur(img, (3,3))
#    img = cv2.resize(img, None, fx = 0.2,fy = 0.2)
#        print(img.shape)