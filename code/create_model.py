# -*- coding: utf-8 -*-
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras.backend as K
from keras.utils import multi_gpu_model
from keras import optimizers
# from keras.utils import plot_model#使用plot_mode时打开
from keras.models import Model
from keras.layers import Conv2D, PReLU, Conv2DTranspose, Add, Concatenate, Input, Dropout, BatchNormalization, Activation, MaxPooling2D,UpSampling2D, Lambda
from keras.layers.core import RepeatVector
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras
from keras import losses
from keras.regularizers import l2
from keras import initializers
from keras.layers.merge import multiply,add


# 戴斯系数 dice = 2*(A交B)/(A并B)
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

# Unet每一层的模块儿单元
def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

# 这里基本就是整个模型了
def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):  #depth为下采样的深度，m为网络输入，inc*dim为网络每一层输出通道数，acti为激活函数，do=dropout,res:是否有res块
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)#下采样过程
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)#上采样过程
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m

def UNet(img_shape=(256,256,5), out_ch=2, start_ch=64, depth=4, inc_rate=2., activation='relu', dropout=0.3, batchnorm=False, maxpool=True, upconv=True, residual=True):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)#输出的维度为256×256×2
    model = Model(inputs=i, outputs=o)

    optimizer_select = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)

    # model.compile(optimizer=optimizer_select, loss=losses.binary_crossentropy, \
    #               metrics=[dice_coef_loss], \
    #               loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

    model.compile(optimizer=optimizer_select, loss=dice_coef_loss, \
                  metrics=[dice_coef_loss], \
                  loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

    return model


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#    os.mkdir('model')
    MODEL_PATH = 'model'
    model = UNet()
    model.summary()
    MODEL_PATH_H5 = os.path.join(MODEL_PATH, 'u_net_33.h5')
    model.save(MODEL_PATH_H5)

    MODEL_PATH_JSON = os.path.join(MODEL_PATH, 'u_net_33.json')
    model_json = model.to_json()
    with open(MODEL_PATH_JSON, "w") as json_file:
        json_file.write(model_json)
    print('Model Generated!')
    model.summary()
