#!/usr/bin/env python
# coding: utf-8


from parser_args import get_parser_with_args
from utils.myUtils import get_logger, myprint, mkdir
from models.ModelManager import ModelManager



def main_train():
    
    #Intialize parameter parser and logger
    arg_parser=get_parser_with_args()
    opt=arg_parser.parse_args()
    opt.result_dir=opt.result_dir+'_'+opt.model
    mkdir(opt.result_dir)
    log_file_name=opt.result_dir+'/'+opt.log_file
    logger=get_logger(log_file_name)
    
    myprint("******************Begin Training************************\n",logger)
    myprint("Begin! Args for training: {} \n".format(opt),logger)
    
    #Training process
    modelManager=ModelManager(opt, logger,pretrained_model_path=None)
    modelManager.train()
    modelManager.predict()
    
    
    
if __name__=='__main__':
    main_train()
    print("The program finished!")