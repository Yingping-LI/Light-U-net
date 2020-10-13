#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from medpy.metric.binary import dc,jc,hd,asd, assd, precision,recall,sensitivity, specificity
import torch.nn as nn
import numpy as np


# In[ ]:


"""
Define the loss function
"""
def get_criterion(opt):
    if opt.loss=='diceloss':
        criterion=DiceLoss()
        
    if opt.loss=='mse':
        criterion=nn.MSELoss()
        
    if opt.loss=='bce':
        criterion=nn.BCELoss()
        
    if opt.loss=='focal':
        criterion=FocalLoss(gamme=0.1)
    
    if opt.loss=='jaccard':
        criterion=jaccard_loss
        
    if opt.loss=='tversky':
        criterion=TverskyLoss(alpha=0.1,beta=0.3)
        
    return criterion
        


# In[ ]:


def compute_scores(preds,labels):
    preds_data=preds.data.cpu().numpy()
    labels_data=labels.data.cpu().numpy()
    
    dice_score=dc(preds_data,labels_data)
    jaccard_coef=jc(preds_data,labels_data)
    hausdorff_dist=hd(preds_data,labels_data)
    asd_score=asd(preds_data,labels_data)
    assd_score=assd(preds_data,labels_data)
    precision_value=precision(preds_data,labels_data)
    recall_value=recall(preds_data,labels_data)
    sensitivity_value=sensitivity(preds_data,labels_data)
    specificity_value=specificity(preds_data,labels_data)
    return {'dice score':dice_score,'jaccard':jaccard_coef,'hausdorff':hausdorff_dist,
           'asd':asd_score,'assd':assd_score,'precision':precision_value,'recall':recall_value,
           'sensitivity':sensitivity_value,'specificity':specificity_value}


def update_metrics(loss,metrics, metric_dict):
    metric_dict["losses"].append(loss.item())
    metric_dict["dice score"].append(metrics['dice score'])
    metric_dict['jaccard'].append(metrics['jaccard'])
    metric_dict['hausdorff'].append(metrics['hausdorff'])
    metric_dict['asd'].append(metrics['asd'])
    metric_dict['assd'].append(metrics['assd'])
    metric_dict['precision'].append(metrics['precision'])
    metric_dict['recall'].append(metrics['recall'])
    metric_dict['sensitivity'].append(metrics['sensitivity'])
    metric_dict['specificity'].append(metrics['specificity'])
    
    return metric_dict
    
def get_mean_metrics(metric_dict):
    return {k: np.mean(v) for k,v in metric_dict.items()}


def initialize_metrics():
    
    metrics={
    'losses':[],
    'dice score':[],
    'jaccard':[],
    'hausdorff':[],
    'asd':[],
    'assd':[],
    'precision':[],
    'recall':[],
    'sensitivity':[],
    'specificity':[]}
    return metrics
    


import torch
import torch.nn as nn
 
class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = - loss.sum() / N
 
		return loss
