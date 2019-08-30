# coding: utf-8


import os
import numpy as np
import nibabel as nib

import keras.backend as K
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras import metrics
from keras import callbacks
from keras.optimizers import Nadam
import random
import tensorflow as tf
from DenseUnet import DenseUNet

THICKNESS = 5
OUTPUT_SIZE = 2
IMAGE_INPUT_SHAPE = [256, 256]
VAL_RATE = 0.2

def dice_coef(y_true, y_pred):
     smooth = 1e-5
     y_true_f = K.flatten(y_true)
     y_pred_f = K.flatten(y_pred)
     intersection = K.sum(y_true_f * y_pred_f)
     return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
"""
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f_0 = K.flatten(y_true[:,:,0])
    y_pred_f_0 = K.flatten(y_pred[:,:,0])
    y_true_f_1 = K.flatten(y_true[:,:,1])
    y_pred_f_1 = K.flatten(y_pred[:,:,1])

    intersection_0 = K.sum(y_true_f_0 * y_pred_f_0)
    intersection_1 = K.sum(y_true_f_1 * y_pred_f_1)

    out_0 = (2. * intersection_0 + smooth) / (K.sum(y_true_f_0 * y_true_f_0) + K.sum(y_pred_f_0 * y_pred_f_0) + smooth)
    out_1 = (2. * intersection_1 + smooth) / (K.sum(y_true_f_1 * y_true_f_1) + K.sum(y_pred_f_1 * y_pred_f_1) + smooth)
    return (out_1+out_0)/2.
"""
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def BCE_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true,y_pred) - K.log(dice_coef(y_true,y_pred))

def window_leveling(img, window_center=0, window_width=400):
    min_ = (2 * window_center - window_width) / 2.0
    max_ = (2 * window_center + window_width) / 2.0
    img = ((img - min_)/(max_ - min_)).astype(np.float32)
    img[img>1]=0.
    img[img<0]=0.

    # img+=0.25     #zero centered
    return img


def data_generator1(data_path, packSize, is_valid_set):

    data_path_ = os.path.join(data_path, 'zhongshan_nii')
    patient_range = os.listdir(os.path.join(data_path_, 'segmentation'))

    if is_valid_set==1:
        patient_range = patient_range[40:]
    else:
        patient_range = patient_range[:40]
    #print(is_valid_set, patient_range)
    
    seg_path = os.path.join(data_path_, 'segmentation')
    volume_path = os.path.join(data_path_, 'volume')

    check = 0  # Count the batch amount of whole Training/Valiation sets
    random.shuffle(patient_range)
    for patient in patient_range:
        select_seg_path = os.path.join(seg_path, patient)
        select_volume_path = os.path.join(volume_path, 'volume' + patient[12:])

        ## Get the selected liver volume
        im = nib.load(select_volume_path)
        liver = np.array(im.get_data())
        liver = window_leveling(liver)

        ## Get the selected mask volume
        mask_ori = nib.load(select_seg_path)
        mask_ori = np.array(mask_ori.get_data())
        mask = (mask_ori == 2) * 1.0  # Mask of lesion
        liver_mask = (mask_ori != 0) * 1.0  # Mask of  liver
        mask_ori = []

        limit = np.shape(liver)[2] - (THICKNESS - 1)  # Index of the last piece
        # To pack the pieces into batches
        for p in range(0, limit):

            ## Get the piece of liver image for input
            liver_piece = liver[::2, ::2, p:p + THICKNESS] * 1.0  # Shrink the size to 256 * 256 to fit

            ## Get the piece of lesion mask for output
            mask_piece = mask[::2, ::2, p + int(THICKNESS / 2)] * 1.0  # Shrink the size to 256 * 256 to fit

            ## Get the piece of liver mask for output
            liver_mask_piece = liver_mask[::2, ::2,
                                p + int(THICKNESS / 2)] * 1.0  # Shrink the size to 256 * 256 to fit

            ## Here only the pieces containing lesions will be used as training or valiation data set
            if sum(sum(np.array(1.0 * (mask_piece == 1)))) != 0:
                check += 1
 
    print(check)
                
def data_generator(data_path, packSize, is_valid_set):

    data_path_ = os.path.join(data_path, 'zhongshan_nii')
    patient_range = os.listdir(os.path.join(data_path_, 'segmentation'))

    if is_valid_set==1:
        patient_range = patient_range[40:]
    else:
        patient_range = patient_range[:40]
    print(is_valid_set, patient_range)
    
    seg_path = os.path.join(data_path_, 'segmentation')
    volume_path = os.path.join(data_path_, 'volume')

    while (1):

        batch_cnt = 0  # Count the batch amount of whole Training/Valiation sets
        random.shuffle(patient_range)
        for patient in patient_range:
            select_seg_path = os.path.join(seg_path, patient)
            select_volume_path = os.path.join(volume_path, 'volume' + patient[12:])

            ## Get the selected liver volume
            im = nib.load(select_volume_path)
            liver = np.array(im.get_data())
            liver = window_leveling(liver)

            ## Get the selected mask volume
            mask_ori = nib.load(select_seg_path)
            mask_ori = np.array(mask_ori.get_data())
            mask = (mask_ori == 2) * 1.0  # Mask of lesion
            liver_mask = (mask_ori != 0) * 1.0  # Mask of  liver
            mask_ori = []

            limit = np.shape(liver)[2] - (THICKNESS - 1)  # Index of the last piece
            check = 0  # To pack the pieces into batches
            for p in range(0, limit):

                ## Get the piece of liver image for input
                liver_piece = liver[::2, ::2, p:p + THICKNESS] * 1.0  # Shrink the size to 256 * 256 to fit

                ## Get the piece of lesion mask for output
                mask_piece = mask[::2, ::2, p + int(THICKNESS / 2)] * 1.0  # Shrink the size to 256 * 256 to fit

                ## Get the piece of liver mask for output
                liver_mask_piece = liver_mask[::2, ::2,
                                   p + int(THICKNESS / 2)] * 1.0  # Shrink the size to 256 * 256 to fit

                ## Here only the pieces containing lesions will be used as training or valiation data set
                if sum(sum(np.array(1.0 * (mask_piece == 1)))) != 0:
                    if check == 0:
                        ## For the first piece, initialze the batch data
                        dat = np.expand_dims(liver_piece, axis=0)
                        # label = np.expand_dims(np.expand_dims(liver_mask_piece, axis=-1),axis=0)
                        label = np.expand_dims(np.concatenate((np.expand_dims(mask_piece, axis=-1), \
                                                               np.expand_dims(liver_mask_piece, axis=-1)), axis=-1),axis=0)
                    else:
                        ## For the other pieces, pack the pieces into batches
                        dat = np.concatenate((dat, np.expand_dims(liver_piece, axis=0)), axis=0)
                        # label = np.concatenate((label, np.expand_dims(np.expand_dims(liver_mask_piece, axis=-1),axis=0)), axis=0)
                        label = np.concatenate(
                            (label, np.expand_dims(np.concatenate((np.expand_dims(mask_piece, axis=-1), \
                                                                   np.expand_dims(liver_mask_piece, axis=-1)), axis=-1),
                                                   axis=0)), axis=0)
                    check += 1
                
                if check == packSize:
                    #yield ({'input_1': dat}, {'conv2d_23': label})
                    yield (dat, label)
                    batch_cnt += 1
                    dat = []
                    label = []
                    check = 0



if __name__ == "__main__":

    ## Select the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## The path of training and valiation sets. Here the training and validation sets share the same path
    ## Before you run this program, please extract the 'raw.tar.gz' file into 'data/MICCAI_2017_LiTS/' folder.
    raw_path = 'raw'
    
    ## The path of U-net model
    model_input_path = 'model'

    ## The saving path of U-net weights
    model_output_path = 'weights'

    ## The logs path of training
    log_path = 'logs-zhongshan'

    ## The batch size. Here I use just 1 because of the poor memory of my laptop...
    packSize = 4
 #   data_generator1(raw_path, packSize, 1)
    ## Load the U-net model, custom_objects={'dice_coef': dice_coef, 'BCE_dice_loss': BCE_dice_loss}
    model_input_path_H5 = os.path.join(model_output_path + '/u_net_33_2-best_wf1.hdf5')
    #model = load_model(model_input_path_H5, custom_objects={'dice_coef_loss': dice_coef_loss})
    #model=load_model(model_input_path_H5,custom_objects={'BCE_dice_loss':dice_loss_loss, 'dice_coef_loss': dice_coef_loss})
    #model=load_model(model_input_path_H5,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef,'BCE_dice_loss': BCE_dice_loss})
    ## Print the structure of U-net if you want. Sometimes you need to check the name of output layer.
    # model.summary()
    model=load_model(model_input_path_H5,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef,'BCE_dice_loss': BCE_dice_loss})
    #model=DenseUNet()
    #model.load_weights('densenet161_weights_tf.h5',by_name=True)
    ## Checkpoint settings
    ## Set the optimizer
    optimizer_select = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

    ## Compile the model
    ## Before you run this program, you should open the 'metrics.py' file of keras package, then add the definition codes
    ##  of 'dice_coef(y_true, y_pred)' function and 'dice_coef_loss(y_true, y_pred)' function after the header commands.
    ## The location of 'metrics.py' file should be somewhere like: '/usr/local/lib/python3.5/dist-packages/keras'
    model.compile(optimizer=optimizer_select, loss=BCE_dice_loss, \
                  metrics=[dice_coef], \
                  loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
   
    check_point = callbacks.ModelCheckpoint(model_output_path + '/u_net_33_zs-best_wf1.hdf5', \
                                            monitor='val_loss', verbose=0, save_best_only=True, \
                                            save_weights_only=False, mode='auto', period=1)

    ## Tensorboard settings
    tensor_board = callbacks.TensorBoard(log_dir=log_path, \
                                         histogram_freq=0, batch_size=1, \
                                         write_graph=True, write_grads=False, \
                                         write_images=False, embeddings_freq=0, \
                                         embeddings_layer_names=None, embeddings_metadata=None)

    #################################################################################
    ## Here, we got:
    ##  Training Dataset Batch Amount: 5623+568  steps_per_epoch=batch_amt_per_training_epoch//packSize
    ##  Validation Dataset Batch Amount: 1535
    ## Accordingly, we set:
    batch_amt_per_training_epoch = 480     #5623
    batch_amt_per_valid_epoch = 114   #1535

    ## Training the model
    model.fit_generator(generator=data_generator(raw_path, packSize, 0), \
                        steps_per_epoch=120,epochs=200, verbose=1, \
                        callbacks=[tensor_board, check_point], \
                        validation_data=data_generator(raw_path, 1, 1), \
                        validation_steps=114, class_weight=None, \
                        max_queue_size=10, workers=1, \
                        use_multiprocessing=False, shuffle=True, initial_epoch=0)
    