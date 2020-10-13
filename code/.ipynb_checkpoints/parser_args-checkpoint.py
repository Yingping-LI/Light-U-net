#!/usr/bin/env python
# coding: utf-8


import argparse 


def get_parser_with_args():
    
    parser=argparse.ArgumentParser(description='Training medical image segmentation network.')
    
    parser.add_argument('-model',
                      type=str,
                      default='base_unet',
                      choices=['base_unet','spectral_unet','deeplab'],
                      required=False,
                      help='choose model for training,for example unet.')
    
    parser.add_argument('-image_size',
                       type=int,
                       default=256,
                       required=False,
                       help='input image size.')
    
    parser.add_argument('-min_epoch',
                      type=int,
                      default=50,
                      required=False,
                      help='at least training min_epoch epochs.')
    
    parser.add_argument('-num_epochs',
                      type=int,
                      default=300,
                      required=False,
                      help='number of the epochs for training.')
    
    parser.add_argument('-batch_size',
                      type=int,
                      default=8,
                      required=False,
                      help='batch size for training')
    
    parser.add_argument('-optimizer',
                      type=str,
                      default='Adam',
                      choices=['Adam','SGD'],
                      required=False,
                      help='Optimizer for training.')
    
    parser.add_argument('-lr',
                      type=float,
                      default=1e-4,
                      required=False,
                      help='learning rate.')
    
    parser.add_argument('-patience',
                      type=int,
                      default=10,
                      required=False,
                      help='stopping training if there is no improvement in these epochs.')
    
    parser.add_argument('-loss',
                      type=str,
                      default='diceloss',
                      choices=['diceloss','bce','smooth_loss'],
                      required=False,
                      help='loss function such as diceloss,bce...')
    
    parser.add_argument('-num_channels',
                       type=int,
                       default=3,
                       required=False,
                       help='number of the input data channels.')

    parser.add_argument('-num_classes',
                       type=int,
                       default=1,
                       required=False,
                       help='number of classes.')
    
    parser.add_argument('-train_dir',
                       type=str,
                       default='../../0)myData/train',
                       required=False,
                       help='data directory for training.')
    
    parser.add_argument('-val_dir',
                       type=str,
                       default='../../0)myData/val',
                       required=False,
                       help='data directory for validation.')
    
    parser.add_argument('-test_dir',
                       type=str,
                       default='../../0)myData/test',
                       required=False,
                       help='data directory for test.')
    
    parser.add_argument('-image_folder',
                       type=str,
                       default='images',
                       required=False,
                       help='image folder name.')
    
    parser.add_argument('-mask_folder',
                       type=str,
                       default='mask',
                       required=False,
                       help='mask folder name.')
    
    parser.add_argument('-GT_folder',
                       type=str,
                       default='GT',
                       required=False,
                       help='Ground Truth folder name.')
    
    parser.add_argument('-contour_folder',
                       type=str,
                       default='contour',
                       required=False,
                       help='contour folder name.')
    
    parser.add_argument('-result_dir',
                       type=str,
                       default='../results',
                       required=False,
                       help='directory to save the results.')
    
    parser.add_argument('-num_workers',
                       type=int,
                       default=20,
                       required=False,
                       help='Number of cpu workers.')
    
    parser.add_argument('-log_file',
                       type=str,
                       default='log.txt',
                       required=False,
                       help='file name to save the logs.')
        
    return parser




