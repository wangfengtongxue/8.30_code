import os 
import pydicom
import numpy as np
import glob
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import nibabel as nib

import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D
from keras.layers import Input, Dense, Add, Concatenate
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras import metrics
from keras import callbacks
from skimage import measure
from tqdm import tqdm
from scipy import ndimage
import tensorflow as tf

def load_scan(path):
    slices = [pydicom.dcmread(s) for s in glob.glob(path + '/*.dcm')]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    print(slice_thickness)
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

# 肝脏的后处理
def liver_post_process(np_array):
    (label_array, num_) = ndimage.label(np_array)
    sum_list = []
    for i in range(num_):
        sum_list.append(np.sum(label_array==i+1))     # 找出体积最大的标记区，舍掉其他的联通域
    index_ = np.argmax(sum_list)
    print(np.unique(label_array), index_)
    label_array[label_array != index_+1] = 0
    label_array[label_array > 1]=1
    return label_array

# 病灶的后处理
def lession_post_process(np_array):
    (label_array, num_) = ndimage.label(np_array)
    for i in range(num_):
        z = np.any(label_array==i+1, axis=(1,2))   # 从z轴看，找出label==i+1的连通域的厚度，小于等于1的直接剔除
        if sum(z)<=1:
            label_array[label_array==i+1]=0
        else:
            continue
    label_array[label_array>1]=1
    return label_array


THICKNESS = 5
OUTPUT_SIZE = 2
IMAGE_INPUT_SHAPE = [256, 256]
VAL_RATE = 0.2

def dice_coef(y_true, y_pred):
    smooth=1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred):
    gamma = 2
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def mixedLoss(y_true, y_pred):
    alpha = 1e-5
    return alpha * focal_loss(y_true,y_pred) - K.log(dice_coef(y_true,y_pred))

def BCE_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true,y_pred) - K.log(dice_coef(y_true,y_pred))

def window_leveling(img, window_center=0, window_width=400):
    min_ = (2 * window_center - window_width) / 2.0
    max_ = (2 * window_center + window_width) / 2.0
    img = ((img - min_)/(max_ - min_)).astype(np.float32)
    img[img>1]=0.
    img[img<0]=0.

    return img



if __name__ == "__main__":

    ## Select the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## The path of training and valiation sets. Here the training and validation sets share the same path
    ## Before you run this program, please extract the 'raw.tar.gz' file into 'data/MICCAI_2017_LiTS/' folder.
    #raw_path = 'data/MICCAI_2017_LiTS/raw/Training_Batch_2'
    raw_path = 'test'

    ## The saving path of U-net weights
    model_output_path = './weights'

    ## The logs path of training
    log_path = 'logs'

    ## The batch size. Here I use just 1 because of the poor memory of my laptop...
    packSize = 1

    ## Load the U-net model
    model_path = os.path.join(model_output_path, 'u_net_33_zs-best.hdf5')  #u_net_weights_save_best_by_loss_50_epoches.hdf5
    #model_path = os.path.join(model_output_path, 'u_net_weights_save_best_by_loss.hdf5')
   #model=load_model(model_path,custom_objects={'BCE_dice_loss':BCE_dice_loss, 'dice_coef': dice_coef_loss})
    model=load_model(model_path,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef,'BCE_dice_loss':BCE_dice_loss})

    for patient in os.listdir(raw_path):
        ct_path = os.path.join(raw_path, patient)
        print(ct_path)
        slices = load_scan(ct_path)
        im = get_pixels_hu(slices)
        print(im.shape)
        # image = np.fliplr(im).transpose(1,2,0).transpose(1,0,2)
        image = im.transpose(2,1,0)
        image = window_leveling(image)

        limit = np.shape(image)[2] - (THICKNESS - 1)  # Index of the last piece
        print(limit)
        check = 0  # To pack the pieces into batches
        ct_3d = []
        liver_3d = []
        lession_3d = []
        for p in tqdm(range(0, limit)):

            ## Get the piece of liver image for input
            liver_piece = image[::2, ::2, p:p + THICKNESS] * 1.0  # Shrink the size to 256 * 256 to fit

            ## Here we use the pieces containing lesions to evaluate the prediction performance
            mask_pred = model.predict_on_batch( np.expand_dims(liver_piece, axis=0) )[0]
            pred_liver = mask_pred[:, :, 1]
            pred_lession = mask_pred[:, :, 0]

            pred_liver[pred_liver > .4] = 1
            pred_liver[pred_liver < .4] = 0

            pred_lession[pred_lession > .5] = 1
            pred_lession[pred_lession < .5] = 0

            ct_3d.append(liver_piece[:,:,2])
            liver_3d.append(pred_liver)
            lession_3d.append(pred_lession)

        liver_3d = liver_post_process(liver_3d)
        lession_3d = lession_post_process(lession_3d)*liver_3d
        for p in tqdm(range(0, limit)):
            liver_piece = ct_3d[p]
            pred_liver = liver_3d[p]
            pred_lession = lession_3d[p]
            fig = plt.gcf()

            plt.imshow(liver_piece, cmap='gray')
            contours = measure.find_contours(pred_liver, 0.1)
            for n, contour in enumerate(contours):
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')
            plt.axis('off')
            fig = plt.gcf()
            fig.gca().xaxis.set_major_locator(plt.NullLocator())
            fig.gca().yaxis.set_major_locator(plt.NullLocator())

            contours = measure.find_contours(pred_lession, 0.1)
            for n, contour in enumerate(contours):
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
            plt.axis('off')
            fig = plt.gcf()
            fig.gca().xaxis.set_major_locator(plt.NullLocator())
            fig.gca().yaxis.set_major_locator(plt.NullLocator())

            path = 'zhongshan_ct_output_post1_1/' +patient+ '/' + str(p) + '.png'
            path1=os.path.split(path)[0]
            if not os.path.exists(path1):
                os.makedirs(path1)
            plt.savefig(path)
            plt.close()

            # plt.figure(figsize=(12,12))
            # plt.subplot(131)
            # plt.imshow(liver_piece[:,:,2])
            # plt.subplot(132)
            # plt.imshow(pred_liver)
            # plt.subplot(133)
            # plt.imshow(pred_lession)
            # path = ct_path.replace('ceshi_zhongshan', 'zhongshan_ct_output_____') + '/' + str(p) + '.png'
            # if not os.path.exists(os.path.split(path)[0]):
            #     os.makedirs(os.path.split(path)[0])
            # plt.savefig(path)
            # plt.close()


