#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
from medpy.metric.binary import dc


"""
Define the loss function.
"""
def get_criterion(opt):
    if opt.loss=='diceloss':
        criterion=DiceLoss()

    elif opt.loss=='bce':
        criterion=nn.BCELoss()
        
    elif opt.loss=='smooth_loss':
        criterion=smooth_loss
    
    else:
        raise Exception('Undifined loss function!')
        
        
    return criterion




"""
dice loss
"""
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def	forward(self, preds, target):
        N = target.size(0)
        #smooth = 0.0001
        smooth = 0

        input_flat = preds.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss




"""
Smooth loss
"""
def smooth_loss(pred, target):
    smooth_weight=0.995
    bce_loss=nn.BCELoss()
    loss=smooth_weight*bce_loss(pred, target)+(1-smooth_weight)*total_variation_loss(pred,target)
    return loss



"""
Total variation loss: to smooth the results
"""
def total_variation_loss(pred, target):
    shape=pred.shape
    img_nrows=256
    img_ncols=256
    
    if len(shape)!=3:
        print('pred image shape=',shape)
    
    h = pred[:, 1:img_nrows, 1:img_ncols] - pred[:, 0:img_nrows-1, 1:img_ncols]
    v = pred[:, 1:img_nrows, 1:img_ncols] - pred[:,  1:img_nrows, 0:img_ncols-1]
    
    h=torch.pow(h,2)
    v=torch.pow(v,2)
    TV=torch.sum(h+v, dim=[1,2])
    TV=torch.sqrt(TV)
    TV=torch.mean(TV)
    
    return  TV