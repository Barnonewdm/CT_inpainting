#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:27:49 2018

@author: wdm
"""

#usage: stack slices into 3D
import SimpleITK as sitk
import numpy as np
from PIL import Image
import sys
from argparse import ArgumentParser
import cv2

def stack_slices(data_dir, tem):
    tem_data = sitk.GetArrayFromImage(tem)
    for i in range(tem_data.shape[0]):
        png = Image.open(data_dir + 'predicted_' + str(i) + '.png')
        png = np.asarray(png)
        save_data = tem_data
        save_data[i,:,:] = cv2.resize(png[::-1,:,0],(tem_data.shape[1],tem_data.shape[2]))
        save_img = sitk.GetImageFromArray(save_data)
        save_img.CopyInformation(tem)
    return save_img

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--tem", type=str, dest="tem",
                        default = './data/liver_image/test/LZH-Prob.nii', help='template 3D image')
    parser.add_argument('--data_dir', type=str, dest='data_dir',
                        default = './data/liver_image/test/predicted/img/')
    parser.add_argument('--save_name', type=str, dest='save_name',
                        default = './data/liver_image/test/predicted/predicted.nii')
    args = parser.parse_args()
    
    tem = sitk.ReadImage(args.tem)
    inpainted_img = stack_slices( data_dir=args.data_dir, tem=tem)
    sitk.WriteImage(inpainted_img, args.save_name)