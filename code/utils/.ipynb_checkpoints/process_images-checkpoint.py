#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
import os
from skimage import io
from .myUtils import *
from PIL import Image
  

    
"""
Function:find the coutour of a given image.
"""
def draw_contour(img,mask):
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)  
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    img=cv2.drawContours(img,contours,-1,(0,0,255),2)
    return img


"""
Function: draw contour for all the images in a folder.
"""
def draw_contours(image_path, mask_path,save_path):
    mkdir(save_path)
    
    file_list=get_filenames(mask_path)    
    for file in file_list:
        basename=os.path.basename(file)
        image_file=os.path.join(image_path,basename)
        save_file=os.path.join(save_path,basename)
        
        image=cv2.imread(image_file)
        mask=cv2.imread(file)
        
        assert image.shape==mask.shape
        
        new_image=draw_contour(image,mask)
        cv2.imwrite(save_file,new_image)
        

        
        
"""
Function: delete the noise of the predict mask(Only save the biggest mask).
"""
def delete_mask_noise(orig_path, dst_path,fill_full=True):
    mask=cv2.imread(orig_path)
    gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)  
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  
    index=0
    if len(contours)>1:
        countour_points=[contours[i].shape[0] for i in range(len(contours))]
        index=countour_points.index(max(countour_points))
    
    if fill_full:
        countour=cv2.drawContours(np.zeros(mask.shape),contours,index,(0,0,255),-1)
    else:
        countour=cv2.drawContours(np.zeros(mask.shape),contours,index,(0,0,255),2)

    new_mask=countour[:,:,2].copy()
    new_mask[new_mask==255]=1
    io.imsave(dst_path,new_mask)
    return new_mask

        

"""
Define the method to get binary mask.
"""
def get_binary_mask(mask):
    mask[mask>0.5]=1
    mask[mask<=0.5]=0

    return mask

