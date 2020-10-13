#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import torch
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from multiprocessing import Pool
import torch.utils.data as data
from utils.myUtils import get_filenames


"""
Define function to read image( for conviently using Pool.map()).
"""
def read_image(path):
    return cv2.imread(path,cv2.IMREAD_COLOR)

def read_gray_image(path):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE)



"""
Data Augmentation.
"""
def sometimes(aug):
    return iaa.Sometimes(0.5,aug)

seq=iaa.Sequential([
    iaa.Fliplr(0.5),
    sometimes(iaa.Affine(scale={'x':(0.8,1.2),'y':(0.8,1.2)},
               translate_percent={'x':(-0.2,0.2),'y':(-0.2,0.2)},
               rotate=(-45,45),
               shear=(-16,16),
               mode=ia.ALL)),
    ],
    random_order=True)
    

"""
Get basenames from the paths.
"""
def get_basenames(path):
    
    file_list=get_filenames(path)
    basename_list=[os.path.basename(file) for file in file_list]
    print('There are {} datas in {}.'.format(len(basename_list), path))
        
    return basename_list


"""
Load the datas.
"""
def load_data(base_path,opt):
    
    basename_images=get_basenames(os.path.join(base_path,opt.image_folder))
    basename_labels=get_basenames(os.path.join(base_path,opt.mask_folder))

    basename_list=list(set(basename_images).intersection(set(basename_labels)))
    print('There are {} images with ground truth.'.format(len(basename_list)))
        
    image_paths=[]
    mask_paths=[]
    for basename in basename_list:
        image_paths.append(os.path.join(base_path, opt.image_folder, basename))
        mask_paths.append(os.path.join(base_path, opt.mask_folder, basename))
            
        
    pool=Pool(opt.num_workers)
    images=pool.map(read_image,image_paths)
    masks=pool.map(read_gray_image,mask_paths)
    pool.close()
        
    datas={}
    for i in range(len(basename_list)):
        datas[basename_list[i]]={'image':images[i],'mask':masks[i]}
        
    return datas,basename_list
            
    
    
"""
Define the Dataset
"""    
class MyDataset(data.Dataset):
    def __init__(self,datas,data_ids,trans,opt,aug):
        self.datas=datas
        self.data_ids=data_ids
        self.transformer=trans
        self.opt=opt
        self.aug=aug
        
    def __getitem__(self,index):
        img_id=self.data_ids[index]
        image=self.datas[img_id]['image']
        mask=self.datas[img_id]['mask']
        
        return self.transformer(img_id,image,mask,self.opt,self.aug)
    
    def __len__(self):
        return len(self.data_ids)

    
"""
The transformation used when loading the images.
"""
def tranformer(img_name,img,mask,opt,aug):
    
    out_img=cv2.resize(img,(opt.image_size,opt.image_size))
    out_mask=cv2.resize(mask,(opt.image_size,opt.image_size))
           
    if aug:
        out_img=np.asarray([out_img])
        out_mask=np.asarray([out_mask])
        out_img,out_mask=seq(images=out_img, segmentation_maps=out_mask)
        out_img=out_img[0]
        out_mask=out_mask[0]
      
    if opt.num_channels==1:
        out_img=out_img[:,:,0]
        out_img=np.expand_dims(out_img,axis=0)
    else:
        out_img=np.transpose(out_img,(2,0,1))
    
    if (np.max(out_img)>1):
        out_img=out_img/255
      
    if (np.max(out_mask)>1):
        out_mask=out_mask/255
        
    out_mask[out_mask>0.5]=1
    out_mask[out_mask<=0.5]=0
    
    return img_name,out_img, out_mask



"""
Define the final dataloader.
"""
def get_data_loader(data_dir,opt,aug=False,shuffle=False):
    
    datas,data_ids=load_data(data_dir,opt)
    dataset=MyDataset(datas,data_ids,tranformer,opt,aug)
    loader=torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=shuffle,num_workers=opt.num_workers)
    
    return loader
 
