#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 01:20:39 2020

@author: kartik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 00:01:24 2019

@author: kartik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:16:37 2019

@author: kartik
"""
import argparse
import numpy as np
from matplotlib import pyplot as plt
import copy
import sys
if  '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path : sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2
import time
import scipy.signal as signal
from sklearn.linear_model import LinearRegression
sys.path.append('/home/kartik/Boeing/hough_transform')
import RefinedHough

#def draw_line(rho,theta,orig_img, line_color = [0,0,255], line_thickness = 20, make_copy = 1):
#    hough_img = copy.deepcopy(orig_img) if make_copy == 1 else orig_img
#    
#    a = np.cos(theta)
#    b = np.sin(theta)
#    x0 = a*rho
#    y0 = b*rho
#    x1 = int(x0 + 10000*(-b))
#    y1 = int(y0 + 10000*(a))
#    x2 = int(x0 - 10000*(-b))
#    y2 = int(y0 - 10000*(a))
#
#    _  = cv2.line(hough_img,(x1,y1),(x2,y2),line_color,line_thickness) 
#    return hough_img
#
#
##finds the lines in the  given image - first normalizes by (41,41) kernel, then finds best line, removes points lying within line and finds best line again
##returns the parameters of the lines
#def find_edges(file_name, num_lines=10, line_thickness = 20, normalization=0,rho_resolution=20,  angle_resolution = 2,visualise =0):
#    img = cv2.imread(file_name)
#    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    img_smooth = cv2.boxFilter(img_grey ,ddepth = -1, ksize = (41,41))
#    canny_ = cv2.Canny(img_smooth ,50,150,apertureSize = 5)
#    
#    
#    #line_num = 0
#    lines_params = []
#    for line_num in range(num_lines):
#        hough_lines = cv2.HoughLines(canny_,rho_resolution,angle_resolution*np.pi/180,1)
#        most_voted_lines = hough_lines[0][0]
#        rho,theta = most_voted_lines
#        lines_params.append((rho,theta))
#        _ = draw_line(rho,theta,canny_, line_color=[0],line_thickness=line_thickness, make_copy=0)
#        
#    if visualise == 1:    
#        for line in lines_params:
#            img = draw_line(line[0],line[1],plt.imread(file_name))
#            plt.imshow(img)
#            plt.show()
#            plt.pause(0.05)
#    
#    return lines_params



def find_local_gradient_points(img_smooth,lines_params,x_start,x_end,x_dist,
                               perp_num_points = 41,perp_dist_bw_points = 2,visualise = 0,img_draw=[]):
    # img_smooth : the image, ideally smoothed out 
    # line_params : the parameters of the line
    # x_start, x_end : the range of the line on which local gradient edges are to be found
    # x_dist : the disctretisation distance between two points where local gradient edges are found
    # perp_num_points : number of points needed on the perpendicular(keep odd)
    # perp_dist_bw_points : the distance to be kept between the pixels sampled
    # img_draw : the image, in original for purpose of visualisation only

    
    (rho,theta) = lines_params ##theta is in radians
#    local_custom_edges = np.zeros_like(img_smooth) ## this will store the points of highest gradiennts alog the perp as 255
    local_custom_edges_x_indices = []
    local_custom_edges_y_indices = []
    
    ##for each point xl,yl on a line (rho,theta) - finds the points with locally highest gradient
    for xl in range(x_start,x_end,x_dist):
        yl = int((-xl*np.cos(theta) + rho)/np.sin(theta))
        perp_theta = theta ##this makes it perpendicular because of the way rest of the image indexing
        perp_m = np.tan(perp_theta)
        point_list = np.zeros((perp_num_points,3)) ##list of points along the perp along with their pixel values
        mid_value = int((perp_num_points-1)/2)
        dist_skips = 0
        for i in range(1,mid_value+1):
            ok = 0
            while(not(ok)): ##due to discretisation, after moving 1 unit distance along the perp line, we may end up at the same place
                i_pixel = i + dist_skips ##i is the number of the point in the array and i_pixel is the actual distance unit being used
                temp = i_pixel*perp_dist_bw_points/(np.sqrt(1+perp_m**2))
                perp_x_1 = xl - temp
                perp_x_2 = xl + temp
                perp_y_1 = int(round(perp_m*(perp_x_1 - xl) + yl) )
                perp_y_2 = int(round(perp_m*(perp_x_2 - xl) + yl))
                perp_x_1 = int(round(perp_x_1))
                perp_x_2 = int(round(perp_x_2))
                perp_pixel_1 = img_smooth[perp_y_1,perp_x_1]
                perp_pixel_2 = img_smooth[perp_y_2,perp_x_2]
            
                ##check if the points selected are already selected before (might happen due to slope and integer discretization)
                prev_x_1,prev_y_1, _ = point_list[mid_value-(i-1),:] 
                if (perp_x_1 == prev_x_1) & (perp_y_1 == prev_y_1): ##means the points are same and we need to move a bit further
                    dist_skips += 1
                else:
                    ok = 1
                    
            point_list[mid_value-i,:] = (perp_x_1,perp_y_1,perp_pixel_1)
            point_list[mid_value+i,:] = (perp_x_2,perp_y_2,perp_pixel_2)
        
        point_list[mid_value,:] = (xl,yl,img_smooth[yl,xl])
        point_list = point_list.transpose()
        perp_pixel_gradient = np.abs(signal.convolve(point_list[2],np.array((-1,-2,-3,0,3,2,1)),mode = 'same'))
        
        ind = np.argpartition(perp_pixel_gradient[3:-3], -3)[-3:]+3 ##indices of top 3 gradients #not considering the ends of the array due to padding
        
        ind = np.argmax(perp_pixel_gradient[3:-3])+3  ##experimenting with taking only one (since anyway using only that)
        perp_grad_x = point_list[0][ind].astype('int') ## x indices of points with highest gradient
        perp_grad_y = point_list[1][ind].astype('int') ## y indices of points with highest gradient
        
#        local_custom_edges[perp_grad_y,perp_grad_x] = 255
        local_custom_edges_x_indices.append(perp_grad_x)
        local_custom_edges_y_indices.append(perp_grad_y)
               
        if visualise == 1:
            _ = cv2.circle(img_draw,(perp_grad_x,perp_grad_y),5,(0,0,255),-1)   
#            _ = cv2.circle(img,(perp_grad_x[1],perp_grad_y[1]),10,155,-1)
#            _ = cv2.circle(img,(perp_grad_x[2],perp_grad_y[2]),10,55,-1)
            
    if visualise == 1:
        orig_line = RefinedHough.draw_line(rho,theta,img_draw, line_color=[255,0,0], line_thickness=10)
        alpha = 1
        final = cv2.addWeighted(img_draw,alpha ,orig_line,(1-alpha),0)
        cv2.namedWindow('local gradient points',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('local gradient points', 1500,1500)
        cv2.imshow('local gradient points',final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#        cv2.imwrite('global_and_local_edge_v4.jpg',final)
            
    return local_custom_edges_x_indices,local_custom_edges_y_indices


##finds the best fit line for the given set of points
# return R2 score, intercept and slope of line
def best_line(x_points,y_points):
    model = LinearRegression().fit(x_points,y_points)
    
    predict = model.predict(x_points)
    return model.score(x_points,y_points),model.intercept_,model.coef_,predict,model


#given the set of x,y points for the local fit line, checks the quality of the edges
# quality is checked by finding consistency of the edge as a window of size n moves along the line

def line_quality_check(x_points, y_points,args, window_size = 25, stride = 10, image=[],visualise = 0,
                       thresh_vert_dist = 10 ,thresh_angle = 3):
    # comparison b/w adjacent windows for detecting missed and broken 
    # quality check of the line within a window for detecting 'unclean' 
    #thresh_vert_dist    dist b/w the y_pred for first x of window(n+1) and y_pred for same x in window(n).in pixels
    #thresh_angle        the angle between the edges in two consecutive windows
    #thresh_90_perc      the thresh dist for 90th percentile to mark as unclean. in pixels
    
    x_points = np.array(x_points).reshape((-1,1))
    y_points = np.array(y_points)
    thresh_90_perc = args.perc_thresh
    prev_score,prev_intercept, prev_slope= -1,-1,-1
    i = 0
    while (i*stride + window_size< len(x_points)):
        x_window = x_points[i*stride:i*stride+window_size]
        y_window = y_points[i*stride:i*stride+window_size]
        score, intercept, slope, y_modelled, model = best_line(x_window,y_window)

        dif = np.abs(y_modelled-y_window)
        error_status = 0
        y_error_pts,x_error_pts = 0,0
        if prev_score == -1:
            prev_score,prev_intercept, prev_slope, prev_model= score,intercept,slope, model
            color = [0,255,0]
        else:
            x_first = x_window[0]
            y_first_pred = model.predict(np.array(x_first).reshape(-1,1))
            prev_y_pred  = prev_model.predict(np.array(x_first).reshape(-1,1))
            
            y_dist = np.abs(y_first_pred-prev_y_pred)[0]
            
            slope_deg = np.degrees(np.arctan(slope))
            prev_slope_deg = np.degrees(np.arctan(prev_slope))
            slope_dif = np.abs(slope_deg - prev_slope_deg)
            print (x_window[0],np.round(np.percentile(dif,90),2),np.round(np.percentile(dif,args.percentile),2),
                   np.round(slope_dif,2), np.round(y_dist,0))
            
            if y_dist>thresh_vert_dist:
               print("dist too much")
               color = [255,0,0]
               error_status=1
            elif slope_dif > thresh_angle: 
                print("angle too much")
                color = [255,0,255]
                error_status=2
#            elif np.percentile(dif,args.percentile) > thresh_90_perc:
#                print("unclean")
#                color = [255,255,255]
#                error_status=3
#                
#                errors = dif>thresh_90_perc
#                y_error_pts = y_window*errors
#                x_error_pts = x_window[:,0]*errors
#                
#                for pt_i in range(window_size):
#                    _ = cv2.circle(image,(x_error_pts[pt_i],y_error_pts[pt_i]),10,[0,0,0],-1)
            else:
                print("ok")
                color = [255,255,0]
            
            prev_score,prev_intercept, prev_slope, prev_model= score,intercept,slope, model
#            print(y_dist,slope_dif)
            
            
        i +=1
        
        _= cv2.line(image,(x_window[0],int(y_modelled[0])),
                               (x_window[window_size-1],int(y_modelled[window_size-1])),color,10)
        if visualise==1:
            if error_status!=0:
                cv2.namedWindow('local',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('local', 900,900)
                cv2.imshow('local',image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if error_status==3:
                    for pt_i in range(window_size):
                        _ = cv2.circle(image,(x_error_pts[pt_i],y_error_pts[pt_i]),5,[0,0,255],-1)
    

##returns the pixel values along the area perpendicular to the given line at point

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--image_num',type=str,help='the image name',default = '0076')
    argparser.add_argument('--edge_num',type=int, help='the edge number to be analysed',default = 2)
    argparser.add_argument('--percentile',type=int,help='the percentile of diff to be checked for unclean', default = 90)
    argparser.add_argument('--perc_thresh',type=int,help='the thresh value for percentile being checked', default = 10)
    argparser.add_argument('--angle_thresh',type=int,help='the thresh value for angle diff', default = 3)
    argparser.add_argument('--vert_dist_thresh',type=int,help='the thresh value for vert distance diff', default =10)

    args = argparser.parse_args()
    print(args)
    
    tic = time.time()

    file_name = '/home/kartik/Boeing/images/Nov 13 - Weird Edges - Multiple Pics/IMG_0071.JPG'
    file_name = '/home/kartik/Boeing/images/Nov 13 - Weird Edges - Multiple Pics/IMG_'+args.image_num+'.JPG'
    ##find the coarse position of the edges using the iterative canny update hough lines
    lines_params = RefinedHough.find_edges(file_name,10, line_thickness=20)
        


    img = cv2.imread(file_name)
    img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    img_smooth = cv2.boxFilter(img_grey ,ddepth = -1, ksize = (5,5))
    img_draw = cv2.imread(file_name)
    
    if args.edge_num==-1:    
        local_edge_x_ind,local_edge_y_ind = find_local_gradient_points(img_smooth, lines_params[0],90,5200,5,
                                                                       perp_num_points=51,visualise=1,img_draw=img_draw)
    else:
        local_edge_x_ind,local_edge_y_ind = find_local_gradient_points(img_smooth, lines_params[args.edge_num],90,5200,5,
                                                                       perp_num_points=51,visualise=1,img_draw=img_draw)
        

    line_quality_check(local_edge_x_ind,local_edge_y_ind ,args,window_size=100, stride=50, image = img_draw, visualise = 1, 
                       thresh_vert_dist=args.vert_dist_thresh, thresh_angle=args.angle_thresh) 
    
    print("Time taken: {}".format(time.time() - tic))

    cv2.namedWindow('final',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('final', 1500,1500)
    cv2.imshow('final',img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
