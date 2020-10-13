#!/usr/bin/env python
# coding: utf-8


from parser_args import get_parser_with_args
from utils.myUtils import get_logger, myprint, mkdir
from models.ModelManager import ModelManager
import time



def main_train():
    startTime = time.time()

    #Intialize parameter parser and logger
    arg_parser=get_parser_with_args()
    opt=arg_parser.parse_args()
    opt.result_dir=opt.result_dir+'_'+opt.model + '_' + opt.optimizer +'_'+opt.loss+'_'+str(opt.lr)+'_('+str(opt.image_size)+','+str(opt.image_size)+')_batsize_'+str(opt.batch_size)+'_weightdecay_'+str(opt.weight_decay)+'_patience_'+str(opt.patience)
    mkdir(opt.result_dir)
    log_file_name=opt.result_dir+'/'+opt.log_file
    logger=get_logger(log_file_name)
    
    myprint("******************Begin Training************************\n",logger)
    myprint("Begin! Args for training: {} \n".format(opt),logger)
    
    #Training process
    modelManager=ModelManager(opt, logger,pretrained_model_path=None)
    modelManager.train()
    endTime = time.time()
    myprint("Training time: {} \n".format(endTime - startTime), logger)

    
    
    
if __name__=='__main__':
    main_train()
    print("The program finished!")