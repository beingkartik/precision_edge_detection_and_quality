#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:48:54 2020

@author: kartik
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
if  '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path : sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pyquaternion as pyq
sys.path.append('/home/kartik/Boeing/')
from LineSegment import LineSegment
from refine_pose import RefinePose
# from RefinedHough import find_edges,draw_line
from hough_transform.RefinedHough import find_edges,draw_line
from scipy.spatial.transform import Rotation

#given 2 lines (in the image-pixel space), intrinsic matrix and the transformation matrix, 
#this returns their intersection(corner) in gloabl coordinates(3D)    
def findGlobalCorner(chamferLine,cubeLine,globalCubeLine ,pose):
    intersectionPoint =  LineSegment.intersectionPt(chamferLine,cubeLine)
    
    chamferLine = LineSegment(chamferLine .start_point,intersectionPoint)    
    cubeLine = LineSegment(cubeLine.start_point,intersectionPoint)
    
    def lossFun(weight,intersectionPoint,line,pose):
        globalPoint = weight*line.start_point + (1-weight)*line.end_point
        candidatePoint = pose.projectPointToImage(globalPoint)
        return pose.pointLoss(candidatePoint,intersectionPoint)
        
    weight = 0
    finalPose = least_squares(lossFun,weight,args = (intersectionPoint,globalCubeLine,pose))
    weight = finalPose.x
    return weight*globalCubeLine.start_point + (1-weight)*globalCubeLine.end_point , intersectionPoint
  
      
if __name__ == "__main__":
    #these are the coordinates of the edges detected in the image 
    #(they may be the segment of the actual edge)    
    # line points as - far coordinate,near coordinate(coordinate towards the intersection)
    
    
    detectedHoughLines = []
#    detectedHoughLines.append(LineSegment((4041,1209),(4663,2250))) #0,1
#    detectedHoughLines.append(LineSegment((4686,2769),(4701,2428))) #2,1
#    detectedHoughLines.append(LineSegment((4487,2907),(2592,3399))) #2,3
    
    import numpy as np
    import os
    image_name = '/home/kartik/Boeing/images/Mar 1/IMG_0310'
    print(image_name)
    file = np.load(image_name+'.npy',allow_pickle=True)
    points = []
    detectedHoughLines = []

    for num,line in enumerate(file):
        rho,theta = line[-3:-1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))
        detectedHoughLines.append(LineSegment((x1,y1),(x2,y2))) 

    file = np.load(image_name+'_chamfer.npy')
    points = []
    chamferLines = []

    for line in file:
        rho,theta = line[-3:-1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))
        chamferLines.append(LineSegment((x1,y1),(x2,y2))) 

    
    intersection_01 = LineSegment.intersectionPt(detectedHoughLines[0],detectedHoughLines[1])
    intersection_12 = LineSegment.intersectionPt(detectedHoughLines[1],detectedHoughLines[2])
    intersection_23 = LineSegment.intersectionPt(detectedHoughLines[2],detectedHoughLines[3])
    intersection_34 = LineSegment.intersectionPt(detectedHoughLines[3],detectedHoughLines[4])
    
    #the lines truncated at the intersection points
    detectedEdges = []
    detectedEdges.append(LineSegment(intersection_01,intersection_12,start_point_extendable=False, end_point_extendable=False))
    detectedEdges.append(LineSegment(intersection_12,intersection_23,start_point_extendable=False, end_point_extendable=False))
    detectedEdges.append(LineSegment(intersection_23,intersection_34,start_point_extendable=False, end_point_extendable=False))
    
    #define the object's coordinates ( in the real world) in global coordinates
    #(theoretically the object remains at the same place in global coordinates 
    # and the camera pose changes)
    
    cadModelLines = []
    cadModelLines.append(LineSegment((0.036,0.026,0.006),(0.036,0.026,-0.006))) #0,4
    cadModelLines.append(LineSegment((-0.036,0.026,-0.006),(0.036,0.026,-0.006))) #2,4
    cadModelLines.append(LineSegment((-0.036,0.026,-0.006),(-0.036,-0.026,-0.006))) # 2,3
    
    #sample points on the object edges
    sampleseed = np.array((0,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1))
    cadModelPoints = []
    cadModelPoints.append(cadModelLines[0].getSamplePoints(len(sampleseed),sampleseed))
    cadModelPoints.append(cadModelLines[1].getSamplePoints(len(sampleseed),sampleseed))
    cadModelPoints.append(cadModelLines[2].getSamplePoints(len(sampleseed),sampleseed))


    factor =1
    fx = 22009.3789563731/factor
    fy = 21844.0598217231/factor
    cx = 2110.94332395329/factor
    cy = 1828.22521846817/factor
        
    transformation_matrix_initial = np.array(((-0.7039103500,-0.7056692200,0.0809515700,-0.01246),
(0.3054733300,-0.1978637600,0.9314202100,0.000855),
(-0.6412545400,0.6803643700,0.3548378300,-0.4308),
(0,0,0,1)))
    pose = RefinePose(fx = fx, fy = fy, cx = cx, cy = cy,
                            transformation_matrix=transformation_matrix_initial )
    
    sol = pose.solveForPose(cadModelPoints,detectedEdges)
    tloss = pose.getLoss(cadModelPoints,detectedEdges)
    print ("Loss: ",tloss)    
    
    
    numLineSegments = len(cadModelPoints) 
    # numLineSegments = 1
    projcadModelPoints = [pose.projectAllToImage(cadModelPoints[x]) for x in range(numLineSegments)]

    for i in range(numLineSegments):
        x,y = zip(*projcadModelPoints[i])
        plt.scatter(x,y)

    corrImagePoints = [pose.getAllCorrespondences(projcadModelPoints[x],detectedEdges[x]) for x in range(numLineSegments)]
    for i in range(numLineSegments):
        x,y = zip(*corrImagePoints[i])
        plt.scatter(x,y)

    image = cv2.imread(image_name+'.JPG')

    chamferUpperLine = chamferLines[0]
    sideEdgeLine = detectedHoughLines[0]
    globalCubeLine = LineSegment((0.036,0.026,0.006),(-0.036,0.026,0.006)) #0,1
    chamferUpperCorner,imagePointU = findGlobalCorner(chamferUpperLine,sideEdgeLine,globalCubeLine ,pose)
    
    #chamfer lower edge
    chamferLowerLine = chamferLines[1]
    sideEdgeLine = detectedHoughLines[5]
    globalCubeLine = LineSegment((-0.036,0.026,-0.006),(-0.036,0.026,0.006))#2,1
    chamferLowerCorner,imagePointL = findGlobalCorner(chamferLowerLine,sideEdgeLine,globalCubeLine ,pose)
    
    chamferWidth = (sum((chamferUpperCorner - chamferLowerCorner)**2))**0.5
    print("Chamfer Width :", chamferWidth)
    cv2.line(image,(chamferUpperLine.start_point[0],chamferUpperLine.start_point[1]),
             (int(imagePointU[0]),int(imagePointU[1])),(255,0,0),10)
    cv2.line(image,(chamferLowerLine.start_point[0],chamferLowerLine.start_point[1]),
             (int(imagePointL[0]),int(imagePointL[1])),(255,0,0),10)
    
    cv2.circle(image,(int(imagePointU[0]),int(imagePointU[1])),40,(0,255,0),-1)
    cv2.circle(image,(int(imagePointL[0]),int(imagePointL[1])),40,(0,255,0),-1)

    p_u=pose.projectPointToImage(chamferUpperCorner)    
    p_l=pose.projectPointToImage(chamferLowerCorner)
    cv2.circle(image,(int(p_u[0]),int(p_u[1])),40,(0,255,0),-1)
    cv2.circle(image,(int(p_l[0]),int(p_l[1])),40,(0,255,0),-1)

    cv2.line(image,(int(imagePointU[0]),int(imagePointU[1])),
             (int(imagePointL[0]),int(imagePointL[1])),(0,150,0),40)

    plt.imshow(image)
    plt.show()
    # cv2.imwrite(image_name+'_results.png',image)
