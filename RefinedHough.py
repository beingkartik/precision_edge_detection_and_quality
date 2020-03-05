#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:05:52 2020

@author: kartik
"""
import copy
import numpy as np
import sys
if  '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path : sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2
from cv2 import imread, imwrite


def draw_line(rho,theta,orig_img, line_color = [0,0,255], line_thickness = 20, make_copy = 1):
    hough_img = copy.deepcopy(orig_img) if make_copy == 1 else orig_img
    
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))

    _  = cv2.line(hough_img,(x1,y1),(x2,y2),line_color,line_thickness) 
#    plt.imshow(hough_img)
#    plt.pause(0.5)
#                print (rho,theta)
#    plt.show()        
    return hough_img

#finds the lines in the  given image - first normalizes by (41,41) kernel, then finds best line, removes points lying within line and finds best line again
#returns the parameters of the lines. If mask is provided, line is found only in the mask
def find_edges(file_name, num_lines=10, line_thickness = 20, normalization=0,rho_resolution=20,  angle_resolution = 2):
    img = cv2.imread(file_name)
    return find_edges_image(img,num_lines,line_thickness,normalization,rho_resolution,angle_resolution)

#same as find_edges , but takes img file instead of filename as input
def find_edges_image(img, num_lines=10, line_thickness = 20, normalization=0,rho_resolution=20,  angle_resolution = 2):
    
    img = img.copy()
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_smooth = cv2.boxFilter(img_grey ,ddepth = -1, ksize = (41,41))
    canny_ = cv2.Canny(img_smooth ,50,150,apertureSize = 5)
    return find_edges_canny(canny_,num_lines,line_thickness,normalization,rho_resolution,angle_resolution)

#same as find_edges , but takes the gradient image(ie canny or something else) instead of filename as input
def find_edges_canny(gradient_img, num_lines=10, line_thickness = 20, normalization=0,rho_resolution=20,  angle_resolution = 2):

    canny_ = gradient_img.copy()
    #line_num = 0
    lines_params = []
    for line_num in range(num_lines):
#        print(line_num)
#        hough_space, max_distance, time_taken = get_normalized_hough_space(canny_,rho_resolution,  angle_resolution)
#        if normalization:
#            semi_gaussian_filter = np.array(((0,0,7,0,0),(0,0,26,0,0),(0,5,41,5,0),(0,0,26,0,0),(0,0,7,0,0)))
#            semi_gaussian_filter = semi_gaussian_filter/(np.sum(semi_gaussian_filter))
#            hough_space = signal.convolve2d(hough_space,semi_gaussian_filter, mode = 'same')
#        
#        most_voted_line = np.argmax(hough_space)
#        x = int(np.floor(most_voted_line/hough_space.shape[1]))
#        y = most_voted_line - x*hough_space.shape[1]
#        rho,theta = (x*rho_resolution - max_distance),np.deg2rad(y*angle_resolution -90)
        hough_lines = cv2.HoughLines(canny_,rho_resolution,angle_resolution*np.pi/180,1)
        print(len(hough_lines))
        most_voted_lines = hough_lines[0][0]
        rho,theta = most_voted_lines
        lines_params.append((rho,theta))
#        img_saver = draw_line(rho,theta,img, line_color=[0,0,255],line_thickness=10, make_copy=1)
        _ = draw_line(rho,theta,canny_, line_color=[0],line_thickness=line_thickness, make_copy=0)
            
#        plt.imsave(folder_name+"/img_line_"+str(line_num)+".png",img_saver)
#        plt.imsave(folder_name+"/canny_line_"+str(line_num)+".png",canny_saver)
        
    return lines_params
