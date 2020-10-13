#!/usr/bin/env python
# coding: utf-8




import os
import cv2
import pydicom 
import logging
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

"""
Function:get all the file names(with the filter) in a directory.
""" 
def get_filenames(path,filter=['.jpg','.png','.bmp','.tif']):
    result=[]
    for path,dir,filelist in os.walk(path):
        for file in filelist:
            afilepath=os.path.join(path,file)
            ext=os.path.splitext(afilepath)[1]
            
            if ext in filter:
                result.append(afilepath)
 
    return result



"""
Function: read a dcm image
"""
def read_dcm_image(path):
    dcm = pydicom.read_file(path)
    array=dcm.pixel_array
    return array

    
    
"""
Function: make a directory.
"""
def mkdir(path):
    isExist=os.path.exists(path)
    if not isExist:
        os.makedirs(path) 
        print(path," is created successfully!")
    else:
        print(path, "exists already!")  
  
    
"""
Function: save a directory.
"""   
def save_dictionary(dic,txt_name):
    file = open(txt_name,'w',encoding='utf-8')
    for key,value in dic.items():
        file.write(key+'\n')
        file.write(str(value)+'\n')
    file.close()
    
    
"""
Function: create a logger to save logs.
"""
def get_logger(log_file_name):
    logger=logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    handler=logging.FileHandler(log_file_name)
    handler.setLevel(logging.DEBUG)
    formatter=logging.Formatter('%(asctime)s: %(name)s (%(levelname)s)  %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


"""
Define a function to print and save log simultaneously.
"""
def myprint(str,logger):
    logger.info(str)
    print(str)


"""
Function: To show the images.
"""
def display_all_results(path_list,titles,sample_nm=None):
    
    List1=get_filenames(path_list[0])
    n=len(List1)
    print('Data num:',n)
    if sample_nm==None:
        random_indexs=[i for i in range(n)]
    else:
        random_indexs=np.random.randint(low=0, high=n, size=sample_nm)
    
    for index in random_indexs:
        img1_file=List1[index]
        basename=os.path.basename(img1_file)
        print('----------------------------------------------------------------------------------------------------------------')
        print(basename)
        subplot_num=len(path_list)
        plt.figure(figsize=(18,18*subplot_num))
        for i in range(subplot_num):
            file=path_list[i]+'/'+basename
            img=cv2.imread(file)
            img=resize(img,(256,256)+(img.shape[2],))
            sub_fig=plt.subplot(1,subplot_num,i+1)
            sub_fig.set_title(titles[i])
            plt.imshow(img)
        
        plt.show()
        

"""
Function: only show images in the basename_list.
"""
def display_results(path_list,titles,basename_list,sample_nm=None):
    
    List1=get_filenames(path_list[0])
    n=len(List1)
    print('Data num:',n)
    if sample_nm==None:
        random_indexs=[i for i in range(n)]
    else:
        random_indexs=np.random.randint(low=0, high=n, size=sample_nm)
    
    for index in random_indexs:
        img1_file=List1[index]
        basename=os.path.basename(img1_file)
        if basename in basename_list:
            print('----------------------------------------------------------------------------------------------------------------')
            print(basename)
            subplot_num=len(path_list)
            plt.figure(figsize=(18,18*subplot_num))
            for i in range(subplot_num):
                file=path_list[i]+'/'+basename
                img=cv2.imread(file)
                img=resize(img,(256,256)+(img.shape[2],))
                sub_fig=plt.subplot(1,subplot_num,i+1)
                sub_fig.set_title(titles[i])
                plt.imshow(img)
        
        plt.show()

