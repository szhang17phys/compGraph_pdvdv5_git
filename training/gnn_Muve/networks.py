#Simplified by Shu, 20250430
#Only for FD1 2000 opchs---


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Dense, concatenate, Multiply, Lambda, Flatten, BatchNormalization, PReLU, ReLU
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np

def outer_product(inputs):
    x, y = inputs
    batchSize = K.shape(x)[0]    
    outerProduct = x[:,:, np.newaxis] * y[:, np.newaxis,:]
    return outerProduct
    


def vkld_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    diff   = (y_true-y_pred)
    loss   = K.abs(K.sum(diff*K.log(y_pred/y_true), axis=-1))
    return loss

#Try square root method for vkld_loss???

# #Suggested by ChatGPT, to fix underestimation of high y_true region---
# #Soft Weighting Based on y_true
# def vkld_loss(y_true, y_pred):
#     y_true = K.clip(y_true, K.epsilon(), 1.0)
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0)

#     kl_term = (y_true - y_pred) * K.log(y_pred / y_true)

#     # Smooth weight: 1.0 + log(y_true + 1e-3), clipped to 1.0 min
#     log_weight = K.log(y_true + 1e-3)
#     boost_weight = 1.0 + K.relu(log_weight)  # ensure weights ≥ 1.0

#     weighted_kl = kl_term * boost_weight
#     return K.abs(K.sum(weighted_kl, axis=-1))




#Network Architecture
#Input: three scalars (Position of scintillation)
#Output: one vecotr (photon detector response)

#===============================================================
#For protoDUNE-VD v5 geometry, containing 40 optical channels---
#Based on previous v4 geometry network---
#Added by Shu, 20250425---
def model_protodunevd_v5(dim_pdr):#dim_pdr: num of opchannels---
    pos_x       = Input(shape=(1,), name='pos_x')
    pos_y       = Input(shape=(1,), name='pos_y')
    pos_z       = Input(shape=(1,), name='pos_z')
    input_layer = [pos_x, pos_y, pos_z]

    # Explicit input normalization clearly stated:
    x_scaled = Lambda(lambda x: (x + 375) / 790, name='x_scaled')(pos_x)
    y_scaled = Lambda(lambda y: (y + 427.4) /854.8, name='y_scaled')(pos_y)
    z_scaled = Lambda(lambda z: (z  + 277.75) / 854.8, name='z_scaled')(pos_z)



    #Part 1: channel 0 - 3 (M-X)
    feat_int_1 = Dense(1)(z_scaled)#intensity layer---
    feat_int_1 = BatchNormalization(momentum=0.9)(feat_int_1)
    feat_int_1 = ReLU()(feat_int_1)
        
    feat_row_1 = Dense(2)(y_scaled)
    feat_row_1 = BatchNormalization(momentum=0.9)(feat_row_1)
    feat_row_1 = ReLU()(feat_row_1)
    
    feat_col_1 = Dense(2)(x_scaled)#intensity layer---
    feat_col_1 = BatchNormalization(momentum=0.9)(feat_col_1)
    feat_col_1 = ReLU()(feat_col_1)
    
    feat_cov_1 = Lambda(outer_product)([feat_row_1, feat_col_1])
    feat_cov_1 = Flatten()(feat_cov_1)
    feat_cov_1 = Multiply()([feat_cov_1, feat_int_1])

    feat_cov_1 = Dense(150)(feat_cov_1)        
    feat_cov_1 = BatchNormalization(momentum=0.9)(feat_cov_1)
    feat_cov_1 = ReLU()(feat_cov_1)


    #Part 2: channel 4 - 11 (C-X)
    feat_int_2 = Dense(1)(x_scaled)
    feat_int_2 = BatchNormalization(momentum=0.9)(feat_int_2)
    feat_int_2 = ReLU()(feat_int_2)
        
    feat_row_2 = Dense(2)(y_scaled)
    feat_row_2 = BatchNormalization(momentum=0.9)(feat_row_2)
    feat_row_2 = ReLU()(feat_row_2)
    
    feat_col_2 = Dense(4)(z_scaled)
    feat_col_2 = BatchNormalization(momentum=0.9)(feat_col_2)
    feat_col_2 = ReLU()(feat_col_2)
    
    feat_cov_2 = Lambda(outer_product)([feat_row_2, feat_col_2])
    feat_cov_2 = Flatten()(feat_cov_2)
    feat_cov_2 = Multiply()([feat_cov_2, feat_int_2])

    feat_cov_2 = Dense(300)(feat_cov_2)        
    feat_cov_2 = BatchNormalization(momentum=0.9)(feat_cov_2)
    feat_cov_2 = ReLU()(feat_cov_2)


    #Part 3: channel 12 - 13 (M-X)
    feat_int_3 = Dense(1)(z_scaled)
    feat_int_3 = BatchNormalization(momentum=0.9)(feat_int_3)
    feat_int_3 = ReLU()(feat_int_3)
        
    feat_row_3 = Dense(2)(y_scaled)
    feat_row_3 = BatchNormalization(momentum=0.9)(feat_row_3)
    feat_row_3 = ReLU()(feat_row_3)
    
    feat_col_3 = Dense(2)(x_scaled)
    feat_col_3 = BatchNormalization(momentum=0.9)(feat_col_3)
    feat_col_3 = ReLU()(feat_col_3)
    
    feat_cov_3 = Lambda(outer_product)([feat_row_3, feat_col_3])
    feat_cov_3 = Flatten()(feat_cov_3)
    feat_cov_3 = Multiply()([feat_cov_3, feat_int_3])

    feat_cov_3 = Dense(100)(feat_cov_3)
    feat_cov_3 = BatchNormalization(momentum=0.9)(feat_cov_3)
    feat_cov_3 = ReLU()(feat_cov_3)


    #Part 4: channel 14 - 17 (PMT)
    feat_int_4 = Dense(1)(z_scaled)
    feat_int_4 = BatchNormalization(momentum=0.9)(feat_int_4)
    feat_int_4 = ReLU()(feat_int_4)
        
    feat_row_4 = Dense(2)(y_scaled)
    feat_row_4 = BatchNormalization(momentum=0.9)(feat_row_4)
    feat_row_4 = ReLU()(feat_row_4)
    
    feat_col_4 = Dense(2)(x_scaled)
    feat_col_4 = BatchNormalization(momentum=0.9)(feat_col_4)
    feat_col_4 = ReLU()(feat_col_4)
    
    feat_cov_4 = Lambda(outer_product)([feat_row_4, feat_col_4])
    feat_cov_4 = Flatten()(feat_cov_4)
    feat_cov_4 = Multiply()([feat_cov_4, feat_int_4])

    feat_cov_4 = Dense(80)(feat_cov_4)
    feat_cov_4 = BatchNormalization(momentum=0.9)(feat_cov_4)
    feat_cov_4 = ReLU()(feat_cov_4)


    #Part 5: channel 18 - 19 (M-X)
    feat_int_5 = Dense(1)(z_scaled)
    feat_int_5 = BatchNormalization(momentum=0.9)(feat_int_5)
    feat_int_5 = ReLU()(feat_int_5)
        
    feat_row_5 = Dense(2)(y_scaled)
    feat_row_5 = BatchNormalization(momentum=0.9)(feat_row_5)
    feat_row_5 = ReLU()(feat_row_5)
    
    feat_col_5 = Dense(2)(x_scaled)
    feat_col_5 = BatchNormalization(momentum=0.9)(feat_col_5)
    feat_col_5 = ReLU()(feat_col_5)
    
    feat_cov_5 = Lambda(outer_product)([feat_row_5, feat_col_5])
    feat_cov_5 = Flatten()(feat_cov_5)
    feat_cov_5 = Multiply()([feat_cov_5, feat_int_5])

    feat_cov_5 = Dense(100)(feat_cov_5)
    feat_cov_5 = BatchNormalization(momentum=0.9)(feat_cov_5)
    feat_cov_5 = ReLU()(feat_cov_5)


    #Part 6: channel 20 - 23 (PMT)
    feat_int_6 = Dense(1)(z_scaled)
    feat_int_6 = BatchNormalization(momentum=0.9)(feat_int_6)
    feat_int_6 = ReLU()(feat_int_6)
        
    feat_row_6 = Dense(2)(y_scaled)
    feat_row_6 = BatchNormalization(momentum=0.9)(feat_row_6)
    feat_row_6 = ReLU()(feat_row_6)
    
    feat_col_6 = Dense(2)(x_scaled)
    feat_col_6 = BatchNormalization(momentum=0.9)(feat_col_6)
    feat_col_6 = ReLU()(feat_col_6)
    
    feat_cov_6 = Lambda(outer_product)([feat_row_6, feat_col_6])
    feat_cov_6 = Flatten()(feat_cov_6)
    feat_cov_6 = Multiply()([feat_cov_6, feat_int_6])

    feat_cov_6 = Dense(80)(feat_cov_6)
    feat_cov_6 = BatchNormalization(momentum=0.9)(feat_cov_6)
    feat_cov_6 = ReLU()(feat_cov_6)


    #Part 7: channel 24 - 29 (PMT)
    feat_int_7 = Dense(1)(z_scaled)
    feat_int_7 = BatchNormalization(momentum=0.9)(feat_int_7)
    feat_int_7 = ReLU()(feat_int_7)
        
    feat_row_7 = Dense(2)(y_scaled)
    feat_row_7 = BatchNormalization(momentum=0.9)(feat_row_7)
    feat_row_7 = ReLU()(feat_row_7)
    
    feat_col_7 = Dense(2)(x_scaled)
    feat_col_7 = BatchNormalization(momentum=0.9)(feat_col_7)
    feat_col_7 = ReLU()(feat_col_7)
    
    feat_cov_7 = Lambda(outer_product)([feat_row_7, feat_col_7])
    feat_cov_7 = Flatten()(feat_cov_7)
    feat_cov_7 = Multiply()([feat_cov_7, feat_int_7])

    feat_cov_7 = Dense(120)(feat_cov_7)
    feat_cov_7 = BatchNormalization(momentum=0.9)(feat_cov_7)
    feat_cov_7 = ReLU()(feat_cov_7)


    #Part 8: channel 30 - 33 (PMT)
    feat_int_8 = Dense(1)(z_scaled)
    feat_int_8 = BatchNormalization(momentum=0.9)(feat_int_8)
    feat_int_8 = ReLU()(feat_int_8)
        
    feat_row_8 = Dense(2)(y_scaled)
    feat_row_8 = BatchNormalization(momentum=0.9)(feat_row_8)
    feat_row_8 = ReLU()(feat_row_8)
    
    feat_col_8 = Dense(2)(x_scaled)
    feat_col_8 = BatchNormalization(momentum=0.9)(feat_col_8)
    feat_col_8 = ReLU()(feat_col_8)
    
    feat_cov_8 = Lambda(outer_product)([feat_row_8, feat_col_8])
    feat_cov_8 = Flatten()(feat_cov_8)
    feat_cov_8 = Multiply()([feat_cov_8, feat_int_8])

    feat_cov_8 = Dense(80)(feat_cov_8)
    feat_cov_8 = BatchNormalization(momentum=0.9)(feat_cov_8)
    feat_cov_8 = ReLU()(feat_cov_8)

    #Part 9: channel 34 - 39 (PMT)
    feat_int_9 = Dense(1)(z_scaled)
    feat_int_9 = BatchNormalization(momentum=0.9)(feat_int_9)
    feat_int_9 = ReLU()(feat_int_9)
        
    feat_row_9 = Dense(2)(y_scaled)
    feat_row_9 = BatchNormalization(momentum=0.9)(feat_row_9)
    feat_row_9 = ReLU()(feat_row_9)
    
    feat_col_9 = Dense(2)(x_scaled)
    feat_col_9 = BatchNormalization(momentum=0.9)(feat_col_9)
    feat_col_9 = ReLU()(feat_col_9)
    
    feat_cov_9 = Lambda(outer_product)([feat_row_9, feat_col_9])
    feat_cov_9 = Flatten()(feat_cov_9)
    feat_cov_9 = Multiply()([feat_cov_9, feat_int_9])

    feat_cov_9 = Dense(120)(feat_cov_9)
    feat_cov_9 = BatchNormalization(momentum=0.9)(feat_cov_9)
    feat_cov_9 = ReLU()(feat_cov_9)


    #combine the nine blocks---
    feat_con = concatenate([feat_cov_1, feat_cov_2, feat_cov_3, feat_cov_4, feat_cov_5, feat_cov_6, feat_cov_7, feat_cov_8, feat_cov_9])
    feat_con = Dense(dim_pdr)(feat_con)
    feat_con = BatchNormalization(momentum=0.9)(feat_con)
    feat_con = ReLU()(feat_con)

    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_con)
    model    = Model(inputs=input_layer, outputs=pdr, name='protodunevd_v5_model')
    
    model.summary()
    #plot_model(model, to_file='./protodunevd_v4.png', show_shapes=True)
    return model