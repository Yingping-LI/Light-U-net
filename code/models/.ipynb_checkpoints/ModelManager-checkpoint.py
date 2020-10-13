#!/usr/bin/env python
# coding: utf-8


import os
from skimage import io,transform 
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import torch.autograd as autograd
import matplotlib.pyplot as plt

from models import *
from utils.loss import get_criterion
from utils.process_images import get_binary_mask,draw_contours,delete_mask_noise
from utils.myUtils import myprint,save_dictionary, mkdir
from utils.evaluation import initialize_metrics,update_eval_metrics,update_loss,get_mean_metrics,calculate_metrics
from data.dataloader import get_data_loader





class ModelManager(object):
    def __init__(self, opt, logger, pretrained_model_path=None):
        super(ModelManager,self).__init__()
        
        self.opt=opt
        self.logger=logger
        self.save_model_path=opt.result_dir+'/'+opt.model+'_'+opt.optimizer+'_lr_'+str(opt.lr)+'_loss_'+opt.loss
        
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        myprint('GPU Avaliable? \n'+str(torch.cuda.is_available())+'\nNumber of GPU:'+str(torch.cuda.device_count()),self.logger)
        
        # Load the model
        if pretrained_model_path==None:
            myprint('Train the model {} from scrath'.format(opt.model),self.logger)
            self.model=self.load_model()
        else:
            myprint('Load pretrained model: {}'.format(pretrained_model_path),self.logger)
            self.model=torch.load(pretrained_model_path)
            
        
        # Load the optimizer and criterion
        self.optimizer= self.get_optimizer()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.criterion=get_criterion(opt)

        self.train_loader=get_data_loader(self.opt.train_dir,self.opt, aug=True,  shuffle=True)
        self.val_loader = get_data_loader(self.opt.val_dir,  self.opt, aug=False, shuffle=False)

    
    
    
    """
    Load models
    """    
    def load_model(self):
        if self.opt.model=='base_unet':
            model=UNet(self.opt.num_channels,self.opt.num_classes).to(self.device)
            
        elif self.opt.model=='spectral_unet':
            model=spetralUNet(self.opt.num_channels,self.opt.num_classes).to(self.device)
            
        elif self.opt.model=='deeplab':
            model=DeepLab_V3plus(backbone='mobilenet', output_stride=16, num_classes=self.opt.num_classes,
                 sync_bn=True, freeze_bn=False).to(self.device)
            
        else:
            raise Exception('Undifined model!')
            
        myprint("Numer of parameters in model {} : {}".format(self.opt.model, sum(p.numel() for p in model.parameters())),self.logger)    
        return model
    
    
    """
    Get optimizers
    """
    def get_optimizer(self):
        if self.opt.optimizer=='Adam':
            optimizer=optim.Adam(self.model.parameters(),lr=self.opt.lr)

        elif self.opt.optimizer=='SGD':
            optimizer=optim.SGD(self.model.parameters(),lr=self.opt.lr)

        else:
            raise Exception('Undifined optimizer!')

        return optimizer
    
    
    """
    Main training process
    """
    def train(self):
        
        TrainingProcessMetrics={}
        ValProcessMetrics={}
        best_metrix = {'val_loss': 1000}

        counting=0
        for epoch in range(1,self.opt.num_epochs+1):   
            myprint("Epoch {}/{}: ".format(epoch,self.opt.num_epochs),self.logger)

            #train
            mean_train_matrix=self.train_one_epoch()
            TrainingProcessMetrics['Epoch_'+str(epoch)]=mean_train_matrix

            #validation
            mean_val_matrix=self.eval(self.val_loader)
            ValProcessMetrics['Epoch_'+str(epoch)]=mean_val_matrix
            
            self.logger.info("Epoch {}/{}: Training:{}\n Validation: {} \n".format(epoch,self.opt.num_epochs,mean_train_matrix,mean_val_matrix))
             
            #update the learning rate
            val_loss = mean_val_matrix['loss']
            self.scheduler.step(val_loss)
            
            #update the image of the training process
            self.plot_training_process(TrainingProcessMetrics,ValProcessMetrics)
            
            #save the best model.
            if mean_val_matrix['loss'] < best_metrix['val_loss']:
                myprint("save weights to {}, performance improved from {:.05f} to {:.05f}".format(self.save_model_path,best_metrix['val_loss'],mean_val_matrix['loss']),self.logger)

                counting=0
                torch.save(self.model,self.save_model_path)
                best_metrix['val_loss'] =mean_val_matrix['loss']
            else:
                counting+=1
                myprint("performance not improved from {:.05f}, counting={}".format(best_metrix['val_loss'] ,counting),self.logger)


            #stop training if performance not improved for long time.
            if counting>=self.opt.patience and epoch>=self.opt.min_epoch:
                myprint("performance not improved for {} epochs, so stop training!".format(counting),self.logger)
                break

                
        #save the training results
        training_process_file=self.opt.result_dir+'/training_process.txt'
        training_process_results={'valiation_metrix':ValProcessMetrics,
                                  'train_metrix':TrainingProcessMetrics}

        save_dictionary(training_process_results,training_process_file)  
        myprint("Finished training!",self.logger)

        
        
    """
    The process of training one epoch.
    """
    def train_one_epoch(self):
        self.model.train()

        i=0
        eval_metrics=initialize_metrics()
        for image_name,batch_img,labels in self.train_loader:
            batch_img=autograd.Variable(batch_img).float().to(self.device)
            labels=autograd.Variable(labels).float().to(self.device)

            self.optimizer.zero_grad()
            preds=self.model(batch_img)
            preds=preds.squeeze()
            labels=labels.squeeze()
            loss=self.criterion(preds,labels)
            loss.backward()
            self.optimizer.step()

            preds=preds.data.cpu().numpy()
            labels=labels.data.cpu().numpy()
            preds=get_binary_mask(preds)
            eval_metrics=update_eval_metrics(preds,labels,eval_metrics)
            eval_metrics=update_loss(loss,eval_metrics)
            mean_eval_metrics=get_mean_metrics(eval_metrics)
            i+=1
            print("\r {}: loss={:.05f} , dice_score={:.05f}".format(i,mean_eval_metrics['loss'],mean_eval_metrics['dice score']),end="")

            #clear batch variables from memory
            del image_name,batch_img, labels
            
        print(" ",end="\n")
        return mean_eval_metrics

    
    
    """
    Evaluate the performance.
    """       
    def eval(self, eval_loader, save_path=None):
        self.model.eval()
        
        if save_path!=None:
            mkdir(save_path['save_pred_mask_path'])
            
        eval_metrics=initialize_metrics()
        with torch.no_grad():
            for image_name,batch_img,labels in eval_loader:
                batch_img=autograd.Variable(batch_img).float().to(self.device)
                labels=autograd.Variable(labels).float().to(self.device)

                preds=self.model(batch_img)
                
                if save_path!=None:
                    self.save_predict_mask(save_path,image_name, preds)
                
                preds=preds.squeeze()
                labels=labels.squeeze()
                loss=self.criterion(preds,labels)
 

                preds=preds.data.cpu().numpy()
                labels=labels.data.cpu().numpy()
                preds=get_binary_mask(preds)
                eval_metrics=update_eval_metrics(preds,labels,eval_metrics)
                eval_metrics=update_loss(loss,eval_metrics)
  
                #clear batch variables from memory
                del image_name,batch_img, labels

        mean_eval_metrics=get_mean_metrics(eval_metrics)        
        print("val: loss={:.05f} , dice_score={:.05f}".format(mean_eval_metrics['loss'],mean_eval_metrics['dice score']))
        return mean_eval_metrics
        
    
    """
    Save predicted mask
    """
    def save_predict_mask(self,save_path, image_name, pred_masks):
        image_path=save_path['image_path']
        save_pred_mask_path=save_path['save_pred_mask_path']
            
        pred_masks=pred_masks.squeeze().data.cpu().numpy()
        for i in range(len(image_name)):
            image_file=os.path.join(image_path, image_name[i])
            predict_mask_file=os.path.join(save_pred_mask_path, image_name[i])
            
            image=io.imread(image_file)
            predict_mask=pred_masks[i,:,:] if len(image_name)!=1 else pred_masks
            predict_mask=transform.resize(predict_mask, (image.shape[0],image.shape[1]))
            predict_mask=get_binary_mask(predict_mask)
            io.imsave(predict_mask_file,predict_mask)


              
    """
    Predict the mask, calculate the dice score, and save the contrast GT image.
    """ 
    def predict_each_dataset(self, base_dir):
        data_loader=get_data_loader(base_dir, self.opt, aug=False, shuffle=False)
            
        base_folder=base_dir.split('/')[-1] 
        
        # save the predicted mask.
        pred_mask_path=os.path.join(self.opt.result_dir,base_folder,'predict_mask')
        save_path={'image_path': os.path.join(base_dir,self.opt.image_folder),
                  'save_pred_mask_path':pred_mask_path}
        self.eval(data_loader, save_path)
        
        #calcute the metrics.
        mask_path=os.path.join(base_dir,self.opt.mask_folder)
        mean_metrics=calculate_metrics(pred_mask_path, mask_path)
        
        #save the contrast image(comparing the ground truth and predict mask).
        GT_path=os.path.join(base_dir,self.opt.GT_folder)
        contrast_GT_path=os.path.join(self.opt.result_dir,base_folder,'contrast_GT')
        draw_contours(GT_path, pred_mask_path,contrast_GT_path)
        
        return mean_metrics
        
        
        
    """
    Main predicting process
    """ 
        
    def predict(self):
        train_metrics=self.predict_each_dataset(self.opt.train_dir)
        val_metrics=self.predict_each_dataset(self.opt.val_dir)
        test_metrics=self.predict_each_dataset(self.opt.test_dir)
        
        #save the metrics
        predict_process_file=os.path.join(self.opt.result_dir,'predict_results.txt')
        predict_process_results={'train_metrix':train_metrics,
                                 'valiation_metrix':val_metrics,
                                 'test_metrix':test_metrics,
                                  }
        
        save_dictionary(predict_process_results,predict_process_file) 
        
        
    """
    Plot losses of the training process
    """    
    def plot_training_process(self,TrainingProcessMetrics,ValProcessMetrics):
        save_fig_path=os.path.join(self.opt.result_dir, 'training_process.jpg')
        
        Epochs=[]
        TrainLosses=[]
        ValLosses=[]
        
        for k,v in TrainingProcessMetrics.items():
            epoch=k.split('_')[1]
            Epochs.append(int(epoch))
            TrainLosses.append(TrainingProcessMetrics[k]['loss'])
            ValLosses.append(ValProcessMetrics[k]['loss'])
        
        plt.clf()
        plt.plot(Epochs,TrainLosses,'r*',label='train_loss')
        plt.plot(Epochs,ValLosses,'b.',label='val_loss')

        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig(save_fig_path)