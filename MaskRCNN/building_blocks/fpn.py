

'''
ARTICLE ON FPN (FEATURE PYRAMID NETWORK): https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c

MODEL RESNET-50: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

FPN: Feature Pyramid Net: This is a way to generate feature maps.
'''


import numpy as np
import tensorflow as tf
import keras.layers as KL
import logging
from MaskRCNN.building_blocks import ops
from MaskRCNN.config import config as conf






def identity_block(x_in, filters, stage, block):
    '''
    No Convolution applied to Shortcut or (layer to be used for skip connection)
    :param x_in:
    :param filters:
    :param strides:
    :param stage:
    :param block:
    :return:
    '''
    f1, f2, f3 = filters

    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'
    relu_name = 'relu' + str(stage) + block + '_branch'

    x_shape = x_in.get_shape().as_list()

    ## BRANCH 2a
    x = ops.conv_layer(x_in, [1, 1, x_shape[-1], f1], stride=1, padding='SAME',
                       scope_name=conv_name + '2a')
    x = ops.batch_norm(x, axis=[0, 1, 2], scope_name=bn_name + '2a')
    x = ops.activation(x, 'relu', relu_name + '2a')
    logging.info('%s: %s', str(conv_name + '2a'), str(x.get_shape().as_list()))

    ## BRANCH 2b
    x = ops.conv_layer(x, [3, 3, f1, f2], stride=1, padding='SAME',
                       scope_name=conv_name + '2b')
    x = ops.batch_norm(x, axis=[0, 1, 2], scope_name=bn_name + '2b')
    x = ops.activation(x, 'relu', relu_name + '2b')
    logging.info('%s: %s', str(conv_name + '2b'), str(x.get_shape().as_list()))

    ## BRANCH 2c
    x = ops.conv_layer(x, [1, 1, f2, f3], stride=1, padding='SAME',
                       scope_name=conv_name + '2c')
    x = ops.batch_norm(x, axis=[0, 1, 2], scope_name=bn_name + '2c')
    logging.info('%s: %s', str(conv_name + '2c'), str(x.get_shape().as_list()))

    ## Add
    x = x + x_in
    x = ops.activation(x, 'relu', relu_name + '_out')
    logging.info('%s: %s', str(relu_name + '_out'), str(x.get_shape().as_list()))

    return x
    
    
    

def conv_block(x_in, filters, strides, stage, block):
    '''
    Convolution applied to Shortcut or (layer to be used for skip connection)
    :param x_in:
    :param filters:
    :param strides:
    :param stage:
    :param block:
    :return:
    '''
    f1, f2, f3 = filters
    
    conv_name = 'res' + str(stage) + block + '_branch'
    bn_name = 'bn' + str(stage) + block + '_branch'
    relu_name = 'relu' + str(stage) + block + '_branch'

    x_shape = x_in.get_shape().as_list()
    
    ## SHORTCUT (Skip Connection)
    shortcut = ops.conv_layer(x_in, [1, 1, x_shape[-1], f3], stride=strides, padding='SAME',
                       scope_name = conv_name + '1')
    shortcut = ops.batch_norm(shortcut, axis=[0, 1, 2], scope_name=bn_name + '1')
    logging.info('%s: %s', str(conv_name + '1'), str(shortcut.get_shape().as_list()))

    ## BRANCH 2a
    x = ops.conv_layer(x_in, [1, 1, x_shape[-1], f1], stride=strides, padding='SAME',
                       scope_name = conv_name + '2a')
    x = ops.batch_norm(x, axis=[0, 1, 2], scope_name=bn_name + '2a')
    x = ops.activation(x, 'relu', relu_name + '2a')
    logging.info('%s: %s', str(conv_name + '2a'), str(x.get_shape().as_list()))

    ## BRANCH 2b
    x = ops.conv_layer(x, [3, 3, f1, f2], stride=1, padding='SAME',
                       scope_name = conv_name + '2b')
    x = ops.batch_norm(x, axis=[0, 1, 2], scope_name=bn_name + '2b')
    x = ops.activation(x, 'relu', relu_name + '2b')
    logging.info('%s: %s', str(conv_name + '2b'), str(x.get_shape().as_list()))

    ## BRANCH 2c
    x = ops.conv_layer(x, [1, 1, f2, f3], stride=1, padding='SAME',
                       scope_name=conv_name + '2c')
    x = ops.batch_norm(x, axis=[0, 1, 2], scope_name=bn_name + '2c')
    logging.info('%s: %s', str(conv_name + '2c'), str(x.get_shape().as_list()))
    
    ## Add
    x = x + shortcut
    x = ops.activation(x, 'relu', relu_name + '_out')
    logging.info('%s: %s', str(relu_name + '_out'), str(x.get_shape().as_list()))
    return x
    
    
def fpn_bottom_up_graph(input_image, resnet_model, stage_5=True):
    '''
    Here we implement a Resnet50 model, and make sure that at every stage we capture the feature map to be used by
    the top-down FPN network. This is required in assistance to predict the feature map.
    
    :param input_image:
    :param stage_5:
    :return:
    '''
    assert resnet_model in ["resnet50", "resnet101"]
    
    h, w = conf.IMAGE_SHAPE[:2]
    logging.info('Image height = %s, width = %s ................', str(h), str(w))
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
    
    
    logging.info('Initiating FPN BOTTOM-UP .................................')
    x = tf.pad(input_image, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
    logging.info('Zero_padded: %s', str(x.get_shape().as_list()))
    
    # STAGE 1
    logging.info('STAGE 1 ...........................')
    x = ops.conv_layer(x, [7,7,3,64], stride=2, padding='VALID', scope_name='conv1')
    x = ops.batch_norm(x, axis=[0, 1, 2], scope_name='bn_conv1')
    x = ops.activation(x, 'relu', 'relu_conv1')
    logging.info('Conv2D: %s', str(x.get_shape().as_list()))
    x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding="SAME")
    logging.info('MaxPool2d: %s', str(x.get_shape().as_list()))
    C1 = x
   
    # STAGE 2
    logging.info('STAGE 2 ...........................')
    x = conv_block(x, filters=[64,64,256], strides=1, stage=2, block='a')
    x = identity_block(x, filters=[64,64,256], stage=2, block='b')
    x = identity_block(x, filters=[64, 64, 256],  stage=2, block='c')
    C2  = x
    
    # STAGE 3
    logging.info('STAGE 3 ...........................')
    x = conv_block(x, filters=[128, 128, 512], strides=2, stage=3, block='a')
    x = identity_block(x, filters=[128, 128, 512], stage=3, block='b')
    x = identity_block(x, filters=[128, 128, 512], stage=3, block='c')
    x = identity_block(x, filters=[128, 128, 512], stage=3, block='d')
    C3 = x

    # STAGE 4
    logging.info('STAGE 4 ...........................')
    x = conv_block(x, filters=[256, 256, 1024], strides=2, stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[resnet_model]
    for i in range(block_count):
        x = identity_block(x, filters=[256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x
    
    # STAGE 5
    logging.info('STAGE 5 ...........................')
    if stage_5:
        x = conv_block(x, filters=[512, 512, 2048], strides=2, stage=5, block='a')
        x = identity_block(x, filters=[512, 512, 2048], stage=5, block='b')
        x = identity_block(x, filters=[512, 512, 2048], stage=5, block='c')
        C5 = x
    else:
        C5 = None

    return [C2,C3,C4,C5]
    
def fpn_top_down_graph(C2, C3, C4, C5):
    '''
    Feature Pyramid Networks: Detecting objects at different scale is difficult, especially time consuming and memory intensive . Here C1,C2,C3,C4,C5 can be thought as feature maps for each stage. They are useful to build the Feature Pyramid Network, each C's are down-sampled at every stage.
    P1, P2, P3, P4, P5 are the feature map layer for prediction
    
    :param C1:
    :param C2:
    :param C3:
    :param C4:
    :param C5:
    :return:
    '''
    logging.info('Initiating FPN TOP-DOWN .................................')
    
    # Feature Map 1
    M5 = ops.conv_layer(C5, [1, 1, C5.get_shape().as_list()[-1], 256], stride=1, padding='SAME', scope_name='fpn_c5p5')  # to reduce the channel depth
    logging.info('FPN - M5: %s', str(M5.get_shape().as_list()))

    # Feature Map 2
    m4_c = ops.conv_layer(C4, [1,1,C4.get_shape().as_list()[-1], 256], stride=1, padding='SAME', scope_name='fpn_c4p4')
    m4_up = KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(M5)
    M4 = KL.Add(name="fpn_p4add")([m4_up, m4_c])
    logging.info('FPN - M4: %s', str(M4.get_shape().as_list()))

    # Feature Map 3
    m3_c = ops.conv_layer(C3, [1, 1, C3.get_shape().as_list()[-1], 256], stride=1, padding='SAME', scope_name='fpn_c3p3')
    m3_up = KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(M4)
    M3 = KL.Add(name="fpn_p3add")([m3_up, m3_c])
    logging.info('FPN - M3: %s', str(M3.get_shape().as_list()))

    # Feature Map 4
    m2_c = ops.conv_layer(C2, [1, 1, C2.get_shape().as_list()[-1], 256], stride=1, padding='SAME', scope_name='fpn_c2p2')
    m2_up = KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(M3)
    M2 = KL.Add(name="fpn_p2add")([m2_up, m2_c])
    logging.info('FPN - M2: %s', str(M2.get_shape().as_list()))
    
    #### CREATE THE FEATURE MAP FOR PREDICTION
    P2 = ops.conv_layer(M2, [3, 3, 256, 256], stride=1, padding='SAME', scope_name='fpn_p2')
    P3 = ops.conv_layer(M3, [3, 3, 256, 256], stride=1, padding='SAME', scope_name='fpn_p3')
    P4 = ops.conv_layer(M4, [3, 3, 256, 256], stride=1, padding='SAME', scope_name='fpn_p4')
    P5 = ops.conv_layer(M5, [3, 3, 256, 256], stride=1, padding='SAME', scope_name='fpn_p5')

    logging.info('FPN - P2 = %s, P3 = %s, P4 = %s, P5 = %s:',
                 str(P2.get_shape().as_list()), str(P3.get_shape().as_list()),
                 str(P4.get_shape().as_list()), str(P5.get_shape().as_list()))
    
    return dict(fpn_c2=C2, fpn_c3=C3, fpn_c4=C4, fpn_c5=C5, fpn_p2=P2, fpn_p3=P3, fpn_p4=P4, fpn_p5=P5)
  
