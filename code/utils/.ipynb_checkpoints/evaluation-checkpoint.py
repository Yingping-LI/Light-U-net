#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import torch.nn as nn
from .myUtils import get_filenames
import matplotlib.pyplot as plt
from medpy.metric.binary import dc,jc,hd,asd, assd, precision,recall,sensitivity, specificity

"""
Initialize the metrics.
"""
def initialize_metrics():
    
    metrics={'loss':[],
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


"""
Update the metrics
"""
def update_eval_metrics(preds,labels,eval_metrics):

    if len(labels.shape)==2:
        preds=np.expand_dims(preds,axis=0)
        labels=np.expand_dims(labels,axis=0)
        
    N = labels.shape[0]
    
    for i in range(N):
        pred=preds[i,:,:]
        label=labels[i,:,:]
        eval_metrics['dice score'].append(dc(pred,label))
        eval_metrics['precision'].append(precision(pred,label))
        eval_metrics['recall'].append(recall(pred,label))
        eval_metrics['sensitivity'].append(sensitivity(pred,label))
        eval_metrics['specificity'].append(specificity(pred,label))


        if np.sum(pred)>0 and np.sum(label)>0:
            eval_metrics['hausdorff'].append(hd(pred,label))
            eval_metrics['asd'].append(asd(pred,label))
            eval_metrics['assd'].append(assd(pred,label))
            eval_metrics['jaccard'].append(jc(pred,label))
        else:
            eval_metrics['hausdorff'].append('nan')
            eval_metrics['asd'].append('nan')
            eval_metrics['assd'].append('nan')
            eval_metrics['jaccard'].append('nan')

    return eval_metrics


"""
Update loss
"""
def update_loss(loss, eval_matrix):
    
    eval_matrix["loss"].append(loss.item())
    return eval_matrix
    
    
"""
Calculate the mean metrics
"""    
def get_mean_metrics(eval_matrix):
    
    mean_eval_matrix={}
    for k, v in eval_matrix.items():
        mean_eval_matrix[k]=np.mean([float(i) for i in v])
    
    return mean_eval_matrix


"""
Calculate the metrics for images in two folders.
"""
def calculate_metrics(pred_path, label_path):
    file_list=get_filenames(pred_path)
    
    eval_matrics=initialize_metrics()
    for file in file_list:
        basename=os.path.basename(file)
        preds=plt.imread(file).astype(np.bool)
        labels=plt.imread(os.path.join(label_path, basename)).astype(np.bool)
        update_eval_metrics(preds,labels,eval_matrics)
        
    mean_metrics=get_mean_metrics(eval_matrics)
    return mean_metrics
    
    
    
    


    



