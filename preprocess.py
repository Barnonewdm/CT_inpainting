#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:28:56 2018

@author: wdm
"""

import cv2
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

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