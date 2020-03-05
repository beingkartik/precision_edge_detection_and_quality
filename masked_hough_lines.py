#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:17:00 2020

@author: kartik
"""

import numpy as np
import sys
import os
if  '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path : sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2
from cv2 import imread, imwrite
from laplacian_pyramid import Image_Handler
import matplotlib.pyplot as plt
sys.path.append('/home/kartik/Boeing/hough_transform')
import RefinedHough

#%% function to save the masking box and line number selected



def maskAndSelectLine(file_name,loading_from_file = False,line_data=None):
    # read image and set mask using Image_handler
    img_handler = Image_Handler(file_name)
    img_handler.setMask()    
    img_grey = img_handler.grey_img
    
    # get canny filter of the image and remove the mask boundary from gradient image
    
    img_smooth = cv2.boxFilter(img_grey ,ddepth = -1, ksize = (21,21))
    plt.imshow(img_smooth)
    plt.pause(0.5) 
    canny_ = cv2.Canny(img_smooth ,50,150,apertureSize = 7)
    plt.imshow(canny_)
    plt.pause(0.5) 
    pts = np.asarray(img_handler.refPt)
    pts = pts.reshape((-1,1,2))
    canny_=cv2.polylines(canny_,[pts], True, color = 0,thickness = 40)
    plt.imshow(canny_)
    plt.pause(0.5)
    lines_params = RefinedHough.find_edges_canny(canny_,num_lines=3)
    if not loading_from_file:

        fig = plt.figure()
        rows = 3
        cols = 4
        mask_pts = np.asarray(img_handler.refPt)
        color_mask = np.zeros_like(img_handler.orig_img)
        color_mask[:,:,0]=img_handler.mask
        color_mask[:,:,1]=img_handler.mask
        color_mask[:,:,2]=img_handler.mask
        for i,line in enumerate(lines_params):
            img = RefinedHough.draw_line(line[0],line[1],img_handler.orig_img) 
            cur_plt = fig.add_subplot(rows,cols,i+1)
            cur_plt.set_title("Line Num : "+str(i))
            plt.imshow(img[min(mask_pts[:,1]):max(mask_pts[:,1]),min(mask_pts[:,0]):max(mask_pts[:,0])])
            
        plt.show()
        plt.waitforbuttonpress()
        plt.close()
        line_num = int(input("Please enter a line num:\n"))
        
        line_data = list(pts.reshape((-1)))
        line_data.extend(lines_params[line_num])
        line_data.append(line_num)
        
        return line_data

    
    #cv2.imshow("image",image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

#%% the main function
if __name__=="__main__":
#    file_name = '/home/kartik/Boeing/images/Feb 21 - Rebuttal/IMG_0187.JPG' 

    file_name = '/home/kartik/Boeing/images/Mar 1/IMG_0310.JPG' 
    all_lines = []
    num_edges = 2
    for i in range(num_edges):
        all_lines.append(maskAndSelectLine(file_name))
        
    np.save(os.path.join(file_name.split(".")[0])+'_chamfer',all_lines)
    
    