#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import transforms, datasets
import torch
import imgaug as ia
import imgaug.augmenters as iaa

from multiprocessing import Pool
import glob
import torch.utils.data as data
from myUtils import get_filenames
import os
import cv2
import numpy as np
import random


def read_image(path):
    return cv2.imread(path,cv2.IMREAD_COLOR)

def read_gray_image(path):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE)

def read_mask(path):
    mask=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    _,mask=cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    return (mask/255.).astype(np.uint8)


"""
Data Augmentation details
"""

def sometimes(aug):
    return iaa.Sometimes(0.5,aug)

seq=iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Fliplr(0.2),
    sometimes(iaa.Affine(scale={'x':(0.8,1.2),'y':(0.8,1.2)},
               translate_percent={'x':(-0.2,0.2),'y':(-0.2,0.2)},
               rotate=(-45,45),
               shear=(-16,16),
               mode=ia.ALL)),
    ],
    random_order=True
)
    
    
def tranformer(img,mask,image_size,aug):
    
    out_img=cv2.resize(img,(image_size,image_size))
    out_mask=cv2.resize(mask,(image_size,image_size))
           
    if aug:
        out_img=np.asarray([out_img])
        out_mask=np.asarray([out_mask])
        ia.seed(random.randint(1, 100))
        out_img=seq.augment_images(out_img)
        out_mask=seq.augment_images(out_mask)
        out_img=out_img[0]
        out_mask=out_mask[0]
        
    out_img=out_img/255
    out_img=np.transpose(out_img,(2,0,1))
    
    return out_img, out_mask
    
    
"""
Load the train/val/test samples.
"""
def load_data(opt):
    AllDatas={}
    AllIndex={}
    for base_path in [opt.train_dir, opt.val_dir, opt.test_dir]:
        file_list=get_filenames(base_path+'/'+opt.image_folders)
        
        basename_list=[]
        image_paths=[]
        mask_paths=[]
        for file in file_list:
            basename=os.path.basename(file)
            basename_list.append(basename)
            image_paths.append(file)
            mask_paths.append(base_path+'/'+opt.mask_folder+'/'+basename)
            
        
        pool=Pool(opt.num_workers)
        images=pool.map(read_image,image_paths)
        masks=pool.map(read_mask,mask_paths)
        pool.close()
        
        
        datas={}
        for i in range(len(basename_list)):
            datas[basename_list[i]]={'image':images[i],'mask':masks[i]}
            
        AllDatas[base_path]=datas
        AllIndex[base_path]=basename_list
        
    return AllDatas[opt.train_dir],AllIndex[opt.train_dir],AllDatas[opt.val_dir],AllIndex[opt.val_dir],AllDatas[opt.test_dir],AllIndex[opt.test_dir]
            
    
    
"""
Define the Dataset
"""    
class MyDataset(data.Dataset):
    def __init__(self,datas,data_ids,trans,image_size,aug):
        self.datas=datas
        self.data_ids=data_ids
        self.transformer=trans
        self.image_size=image_size
        self.aug=aug
        
    def __getitem__(self,index):
        img_id=self.data_ids[index]
        
        image=self.datas[img_id]['image']
        mask=self.datas[img_id]['mask']
        return self.transformer(image,mask,self.image_size,self.aug)
    
    def __len__(self):
        return len(self.data_ids)



"""
Define the final dataloaders.
"""
def get_loader_singleimage(opt):
    
    train_datas,train_ids,val_datas,val_ids,test_datas,test_ids=load_data(opt)
    
    
    train_dataset=MyDataset(datas=train_datas, data_ids=train_ids,trans=tranformer,image_size=opt.image_size,aug=opt.aug)
    val_dataset=MyDataset(datas=val_datas, data_ids=val_ids,trans=tranformer,image_size=opt.image_size,aug=False)
    test_dataset=MyDataset(datas=test_datas, data_ids=test_ids,trans=tranformer,image_size=opt.image_size,aug=False)
    
    train_loader=torch.utils.data.DataLoader(train_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.num_workers)
    
    val_loader=torch.utils.data.DataLoader(val_dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=False,
                                          num_workers=opt.num_workers)
    
    test_loader=torch.utils.data.DataLoader(test_dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=False,
                                            num_workers=opt.num_workers)
    
    return train_loader,val_loader,test_loader




def get_data_loader(opt):
    return get_loader_singleimage(opt)




