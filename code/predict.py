#!/usr/bin/env python
# coding: utf-8

import os
from parser_args import get_parser_with_args
from utils.myUtils import get_logger, myprint,mkdir
from models.ModelManager import ModelManager



def main_predict():
    
    #Intialize parameter parser and logger
    arg_parser=get_parser_with_args()
    opt=arg_parser.parse_args()
    opt.result_dir=opt.result_dir+'_'+opt.model + '_' + opt.optimizer +'_'+opt.loss+'_'+str(opt.lr)+'_('+str(opt.image_size)+','+str(opt.image_size)+')_batsize_'+str(opt.batch_size)+'_weightdecay_'+str(opt.weight_decay)+'_patience_'+str(opt.patience)
    mkdir(opt.result_dir)
    log_file_name=opt.result_dir+'/'+opt.log_file
    logger=get_logger(log_file_name)
    
    myprint("******************Begin Predicting************************\n",logger)
    myprint("Begin! Args for Predicting: {} \n".format(opt),logger)
    
    #Load model manager
    pretrained_model_path=opt.result_dir+'/'+opt.model+'_'+opt.optimizer+'_lr_'+str(opt.lr)+'_loss_'+opt.loss
    modelManager=ModelManager(opt, logger,pretrained_model_path)
    
    #predict.
    modelManager.predict()

    
    
    
if __name__=='__main__':
    main_predict()
    print("The program finished!")