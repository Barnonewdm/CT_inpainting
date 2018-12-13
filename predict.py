'''
Name: Predict function for inpainting
author: dongming.wei@sjtu.edu.cn 
date: 2018.12.12
usage: python3.6 --save_name ./data/liver_image/test/predicted.png 
         --input_ori_image ./data/liver_image/test/ct/img/test_44.png
         --input_mask_image ./data/liver_image/test/ct/mask/seg_44.png
'''
from skimage import io
from copy import deepcopy
import numpy as np
import os
from libs.pconv_model import PConvUnet
from matplotlib import pyplot as plt
#from libs.util import random_mask_line_for_ct
import sys
from argparse import ArgumentParser
from keras.preprocessing.image import ImageDataGenerator
import glob

class DataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:
            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample
#             mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)
            #mask = np.stack([random_mask_line_for_ct(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)  
            #for _ in range(ori.shape[0], axis=0)
            mask = io.imread(args.input_mask_name)
            if mask.shape[2] != 3:
                mask = mask[:, :, 0:3]
            print('mask shape {}'.format(mask.shape))
            print('mask_max {}'.format(mask.max()))
            mask = mask/255.
            mask = 1 - mask
            print ('original max {}'.format(ori.max()))
            #ori = ori / ori.max()
            ori = ori / 255.
            
            masked = deepcopy(ori)
            masked[mask==mask.min()] = 1
            print("mask shape {}".format(masked.shape))
            
            ori = np.expand_dims(ori, axis=0)
            
            mask = np.uint8(np.expand_dims(mask, axis=0))
            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    test_path = "./data/liver_image/test/ct/"
    parser = ArgumentParser()
    parser.add_argument("--save_name", type=str, dest="save_name",
                        default='./data/liver_image/test/predicted.png', help='save dir and name')
    parser.add_argument("--input_ori_name", type=str, dest="input_ori_name",
                        default = test_path + 'img/test_44.png')
    parser.add_argument("--input_mask_name", type=str, dest="input_mask_name",
                        default = test_path + 'mask/seg_44.png')
    parser.add_argument("--input_dir", type=str, dest="input_dir",
                        default = test_path)
    args = parser.parse_args()
    
    TEST_DIR=args.input_dir
    test_datagen = DataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            TEST_DIR, target_size=(512, 512), batch_size=1, seed=1, shuffle=False,
            )
    '''
    #ori = io.imread(test_path + 'img/' + 'test_'+ sys.argv[1] +'.png')
    ori = io.imread(args.input_ori_name)
    print('original shape {}'.format(ori.shape))
    #mask = io.imread(test_path + 'mask/' + 'seg_'+ sys.argv[1] +'.png')
    mask = io.imread(args.input_mask_name)
    if mask.shape[2] != 3:
        mask = mask[:, :, 0:3]
    print('mask shape {}'.format(mask.shape))
    print('mask_max {}'.format(mask.max()))
    mask = mask/255.
    mask = 1 - mask
    print ('original max {}'.format(ori.max()))
    #ori = ori / ori.max()
    ori = ori / 255.
    
    masked = deepcopy(ori)
    masked[mask==mask.min()] = 1
    print("mask shape {}".format(masked.shape))
    
    ori = np.expand_dims(ori, axis=0)
    
    mask = np.uint8(np.expand_dims(mask, axis=0))
    masked = np.expand_dims(masked, axis=0)
    
    #mask = np.stack([random_mask_line_for_ct(ori.shape[1],ori.shape[2]) for _ in range(ori.shape[0])], axis=0)
    #masked = deepcopy(ori)
    masked[mask==0]=1
    
    model = PConvUnet(weight_filepath='data/logs_ct_new/')
    model.load("data/logs_ct_new/316_weights_2018-12-11-10-45-45.h5")
    pred_img = model.predict([masked, mask])
    
    
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(ori[0])
    axes[1].imshow(masked[0])
    axes[2].imshow(pred_img[0] * 1.)
    axes[0].set_title('Original Image')
    axes[1].set_title('Masked Image')
    axes[2].set_title('Predicted Image')
    plt.savefig(r'data/liver_image/test/img_test_result.png')
    plt.imsave(args.save_name,pred_img[0]*1.)
    #plt.show()
    #plt.close()
    '''
    mask_list = sorted(glob.glob(args.input_dir + 'mask/*'))
    img_list = sorted(glob.glob(args.input_dir + 'img/*'))
    if len(mask_list) != len(img_list):
        print('[Error]: test dataset error')
    else:
        ori = io.imread(img_list[1])
        mask = io.imread(mask_list[1])
        if mask.shape[2] != 3:
                mask = mask[:, :, 0:3]
        masks = np.zeros((len(mask_list),mask.shape[0], mask.shape[1], mask.shape[2]))
        maskeds = np.zeros((len(img_list),ori.shape[0], ori.shape[1], ori.shape[2]))
        for i, f in enumerate(mask_list):
            ori = io.imread(img_list[i])
            print('original shape {}'.format(ori.shape))
            #mask = io.imread(test_path + 'mask/' + 'seg_'+ sys.argv[1] +'.png')
            mask = io.imread(f)
            if mask.shape[2] != 3:
                mask = mask[:, :, 0:3]
            print('mask shape {}'.format(mask.shape))
            print('mask_max {}'.format(mask.max()))
            mask = mask/255.
            mask = 1 - mask
            print ('original max {}'.format(ori.max()))
            #ori = ori / ori.max()
            ori = ori / 255.
            
            masked = deepcopy(ori)
            masked[mask==mask.min()] = 1
            print("mask shape {}".format(masked.shape))
            
            ori = np.expand_dims(ori, axis=0)
            
            mask = np.uint8(np.expand_dims(mask, axis=0))
            masked = np.expand_dims(masked, axis=0)
            
            #mask = np.stack([random_mask_line_for_ct(ori.shape[1],ori.shape[2]) for _ in range(ori.shape[0])], axis=0)
            #masked = deepcopy(ori)
            masked[mask==0]=1
            masks[i,] = mask
            maskeds[i,] = masked
            
        model = PConvUnet(weight_filepath='data/logs_ct_new/')
        model.load("data/logs_ct_new/316_weights_2018-12-11-10-45-45.h5")
        
        pred_img = model.predict([maskeds, masks])
        for i in range(len(img_list)):
            print("saveing" + str(i))
            split = img_list[i]
            splited = split.split('test_')[-1]
            plt.imsave(args.save_name + splited,pred_img[i]*1.)
            