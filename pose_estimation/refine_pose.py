#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 02:28:45 2020

@author: kartik
"""

import numpy as np
import pyquaternion as pyq
from scipy.optimize import least_squares
from scipy.optimize import minimize   
from scipy import optimize
from scipy.spatial.transform import Rotation

#this class is responsible for all the work
#initialise with the intrinsic camera coordinates, initial pose estimate may also be defined
#just call the solveforPose() to solve for the pose


# totalLoss function defines the projection Loss given a set of 3D points, pose and corresponding lines in the Image
class RefinePose():
    
    ##Define the camera intrinsic Matrix
    def __init__(self,fx = 10,fy = 10,cx = 100,cy = 100,transformation_matrix=np.identity(4)):
        
        #fx,fy are expressed in pixels 
        #: fx = Fx * s , where Fx is focal length in world units, s is number of pixels per world unit
        #cx,cy are expressed in pixels
        
        self.camera_intrinsic = np.array(((fx,0,cx,0),(0,fy,cy,0),(0,0,1,0)),dtype = np.float32)
        self.transformation_matrix = transformation_matrix
        
    #return a set of points,P such that P[i] is a point on line that is closest to setPoints[i] 
    def getAllCorrespondences(self,setPoints,line):       
        corrPoints = []
        for pt in setPoints:
            corrPt = line.perpPointLineSegment(pt)
            corrPoints.append(corrPt)
            
        return corrPoints
    
    #given a #D point, returns the corresponding image pixel coordinates
    def projectPointToImage(self,point):
        projMatrix = np.dot(self.camera_intrinsic, self.transformation_matrix)
        homoPoint = np.ones(4)
        homoPoint[0:3] = point
        projPoint = np.dot(projMatrix,homoPoint)
        return np.array((projPoint[0:2]/projPoint[2]))

    #given a set of 3D points, this returns the corresponding image pixel coordinates
    def projectAllToImage(self,setPoints):
        projPoints = []
        for pt in setPoints:
            projPoints.append(self.projectPointToImage(pt))
        
        return projPoints
    
    ##Find the euclidian distance between 2 points
    def pointLoss(self, estPoint, gtPoint): 
        if (estPoint is None or gtPoint is None):
            return 1000
        loss = (estPoint[0]-gtPoint[0])**2 + (estPoint[1] - gtPoint[1])**2
        return loss

    def getLoss(self,cadModelPoints,detectedEdges):
        numLineSegments = len(cadModelPoints)        
        # numLineSegments = 1
        projcadModelPoints = [self.projectAllToImage(cadModelPoints[x]) for x in range(numLineSegments)]
        corrImagePoints = [self.getAllCorrespondences(projcadModelPoints[x],detectedEdges[x]) for x in range(numLineSegments)]
        
        total_loss = 0
        for i in range(numLineSegments):
            for j in range(len(cadModelPoints[i])):
                total_loss += self.pointLoss(projcadModelPoints[i][j],corrImagePoints[i][j])
        
        return total_loss/2
    
    def totalLoss_trans(self,trans,cadModelPoints,detectedEdges):
        transformation_matrix_body = np.identity(4)
        transformation_matrix_body[:3,3] = trans
        
        transformation_matrix_orig = self.transformation_matrix.copy()
        self.transformation_matrix = transformation_matrix_body @ self.transformation_matrix
        loss = self.getLoss(cadModelPoints,detectedEdges)
        self.transformation_matrix = transformation_matrix_orig.copy()
        
        return loss

    def totalLoss_trans_iter(self,val,i,trans,cadModelPoints,detectedEdges):
       trans[i] =val
       return self.totalLoss_trans(trans,cadModelPoints,detectedEdges)

    def totalLoss_rot(self,euler,cadModelPoints,detectedEdges):
        r = Rotation.from_euler('XYZ',euler)
        transformation_matrix_body = np.identity(4)
        transformation_matrix_body[:3,:3] = r.as_matrix()
        
        transformation_matrix_orig = self.transformation_matrix.copy()
        self.transformation_matrix = self.transformation_matrix @ transformation_matrix_body
        loss = self.getLoss(cadModelPoints,detectedEdges)
        self.transformation_matrix = transformation_matrix_orig.copy()
        
        return loss
    
    def totalLoss_rot_iter(self,val,i,euler,cadModelPoints,detectedEdges):
        euler[i] = val
        return self.totalLoss_rot(euler,cadModelPoints,detectedEdges)
        
    def g1(self,x):
        return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 1
    
    def solveForPose(self,cadModelPoints,detectedEdges):

        loss = self.getLoss(cadModelPoints,detectedEdges)
        while(loss>10):
            trans = np.zeros(3)
            euler = np.zeros(3)
            transformation_matrix_orig = self.transformation_matrix.copy()
            for i,val in enumerate(trans):
                bounds = (-0.01,0.01)
                # finalPose = least_squares(self.totalLoss_trans_iter,val,args = (i,trans,cadModelPoints,detectedEdges),bounds=bounds,loss='cauchy')
                finalPose = minimize(self.totalLoss_trans_iter,val,args = (i,trans,cadModelPoints,detectedEdges),method='Nelder-Mead')
                trans[i]  = finalPose.x
                
            transformation_matrix_body = np.identity(4)
            transformation_matrix_body[:3,3] = trans
    
            self.transformation_matrix = transformation_matrix_body @ transformation_matrix_orig
            transformation_matrix_orig = self.transformation_matrix.copy()
            for i,val in enumerate(euler):
                bounds = (-3.14/10,3.14/10)
                # finalPose = least_squares(self.totalLoss_trans_iter,val,args = (i,euler,cadModelPoints,detectedEdges), bounds=bounds,loss='soft_l1')
                finalPose = minimize(self.totalLoss_rot_iter,val,args = (i,euler,cadModelPoints,detectedEdges),method='Nelder-Mead')
                euler[i]  = finalPose.x
                
            r = Rotation.from_euler('XYZ',euler)
            transformation_matrix_body = np.identity(4)
            transformation_matrix_body[:3,:3] = r.as_matrix()
            self.transformation_matrix = transformation_matrix_orig @ transformation_matrix_body
        
            loss = self.getLoss(cadModelPoints,detectedEdges)
            print(loss)
            
                
        return [euler,trans]
 