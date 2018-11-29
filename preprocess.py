#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:28:56 2018

@author: wdm
"""

import cv2
import numpy as np
import SimpleITK as sitk
from argparse import ArgumentParser
#from matplotlib import pyplot as plt

'''
img = sitk.GetArrayFromImage(sitk.ReadImage('./data/liver_image/test/LZH-Prob.nii'))

for i in range(img.shape[0]):
    print ('slice: %d' %i)
    a = img[i,:,:]
    a = a[::-1]
    a = np.expand_dims(a,2)
    a = np.concatenate((a,a,a),2)
    a = a - a.min()
    a = a / a.max()
    a = a*255.
    a = cv2.resize(a, (512,512))
    cv2.imwrite('./data/liver_image/test/ct/img/LZH_ct_%d.png'%i, a)


mask = sitk.GetArrayFromImage(sitk.ReadImage('./data/liver_image/test/LZH-Seg.nii.gz'))

for i in range(mask.shape[0]):
    print ('slice: %d' %i)
    a = mask[i,:,:]
    a = np.float32(a)
    a = a - mask.min()
    a = a/mask.max()
    a = a[::-1]*255.
    a = np.expand_dims(a,2)
    a = np.concatenate((a,a,a),2)
    a = cv2.resize(a, (512,512))
    #a = np.float32(a)
    #plt.imsave('./data/liver_image/test/ct/mask/LZH_ct_%d.png'%i,a)
    cv2.imwrite('./data/liver_image/test/ct/mask/LZH_ct_%d.png'%i,a)
''' 
def threscut(subject_data, threshold_min=-500, threshold_max=500):
    subject_data[subject_data>500]=500
    subject_data[subject_data<-500]=-500
    
    return subject_data

def normalize(slice_i, nor_min=-500, nor_max=500):
    slice_i = np.float32(slice_i)
    slice_i = slice_i - nor_min
    slice_i = slice_i / nor_max

    return slice_i
    
if __name__ ==  "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--nii_path", type=str, 
                        dest="nii_path", default='XXX',
                        help="training, validation (test) data path")
    parser.add_argument("--test_subject", type=str,
                        dest="test_subject", default="XXX")
    parser.add_argument("--save_path", type=str,
                        dest="save_path", default='XXX',
                        help="save path")
    args = parser.parse_args()
    print ("[Info]: processing "+ args.nii_path)
    
    subject = sitk.ReadImage(args.nii_path)
    subject_data = sitk.GetArrayFromImage(subject)
    threshold = 500
    nor=threshold
    
    subject_data = threscut(subject_data, -threshold, threshold)
    for i in range(subject_data.shape[0]-2):
        slice_i = subject_data[i,::-1,:]
        slice_i = normalize(slice_i)
        slice_i = np.expand_dims(slice_i, axis=2)
        slice_i = cv2.resize(slice_i, (512,512))
        
        slice_i_1 = subject_data[i+1,::-1,:]
        slice_i_1 = normalize(slice_i_1)
        slice_i_1 = np.expand_dims(slice_i_1, axis=2)        
        slice_i_1 = cv2.resize(slice_i_1, (512,512))
        
        slice_i_2 = subject_data[i+2,::-1,:]
        slice_i_2 = normalize(slice_i_2)
        slice_i_2 = np.expand_dims(slice_i_2, axis=2)        
        slice_i_2 = cv2.resize(slice_i_2, (512,512))
        
        slice_rgb = np.concatenate((slice_i, slice_i_1, slice_i_2), axis=-1)
        cv2.imwrite(args.save_path + "/{}_ct_{}.png".format(1,i), slice_rgb)
        