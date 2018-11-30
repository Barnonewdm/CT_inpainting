#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:32:35 2018

@author: wdm
"""
import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib

def threscut(subject_data, threshold_min=-500, threshold_max=500):
    subject_data[subject_data>500]=500
    subject_data[subject_data<-500]=-500
    
    return subject_data

def normalize(slice_i, nor_min=-500, nor_max=500):
    slice_i = np.float32(slice_i)
    slice_i = slice_i - nor_min
    slice_i = slice_i /np.float32( nor_max - nor_min)

    return slice_i

img = sitk.ReadImage('./data/liver_image/test/LZH-Prob.nii')
img_data = sitk.GetArrayFromImage(img)
mask = sitk.ReadImage('./data/liver_image/test/LZH-Seg2.nii.gz')
mask_data = sitk.GetArrayFromImage(mask)
[slice_index] = np.where(np.sum(mask_data, axis=tuple([1,2]))!=0)
print slice_index

for i in slice_index:
    mask_tmp = np.float32(mask_data[i,::-1,:])
    img_data_t = threscut(img_data)
    img_tmp = np.float32(img_data_t[i,:,:])
    img_tmp = normalize(img_tmp)
    img_tmp = img_tmp[::-1]
    img_tmp = cv2.resize(img_tmp,(512,512))*255
    img_tmp = np.expand_dims(img_tmp,axis=2)
    img_tmp = np.concatenate((img_tmp,img_tmp,img_tmp),axis=-1)
    cv2.imwrite('./data/liver_image/test/ct/img/test_{}.png'.format(i), img_tmp, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
    mask_tmp = cv2.resize(mask_tmp,(512,512))
    mask_tmp = np.expand_dims(mask_tmp, axis=2)
    mask_rgb = np.concatenate((mask_tmp,mask_tmp,mask_tmp), axis=-1)
    matplotlib.image.imsave('./data/liver_image/test/ct/mask/seg_{}.png'.format(i),mask_rgb)