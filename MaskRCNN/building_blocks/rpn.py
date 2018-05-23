'''

As discussed in the notes section, RPN will have two outputs,
    1) Classify a pixel point as foreground or background
    2) Classify the anchor and give a bounding box for it. For example if we are trying to evaluate 9 anchors,
    then the output at each pixel point would be 9x4 = 36
'''


import logging
import tensorflow as tf
from MaskRCNN.building_blocks import ops
from MaskRCNN.config import config as conf


def get_pixel_fb_classification(x, anchor_stride, anchor_per_location):
    '''
    Get the pixel classification of foreground and background
    :return:
    '''
    sh_in = x.get_shape().as_list()[-1]
    
    # Here 2*anchor_per_location = 6, where 2 indicates the binary classification of Foreground and background and anchor_per_location = 3
    x = ops.conv_layer(x, k_shape=[1, 1, sh_in, 2 * anchor_per_location], stride=anchor_stride, padding='VALID', scope_name='rpn_class_raw')
    logging.info('RPN - Conv Class: %s', str(x.get_shape().as_list()))
    
    # Here we convert {anchor_per_location = 3}
    # [batch_size, h, w, num_anchors] to [batch_size, h*w*anchor_per_location, 2]
    # For each image, at each pixel classify 3 anchors as foreground or background
    rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
    logging.info('rpn_class_logits: %s', rpn_class_logits.get_shape().as_list())


    # Do a softmax classificaion to get output probabilities
    rpn_probs = tf.nn.softmax(rpn_class_logits, name='rpn_class_xxx')
    logging.info('rpn_probs: %s', rpn_probs.get_shape().as_list())
    
    return rpn_class_logits, rpn_probs

def get_bounding_box(x, anchor_stride, anchor_per_location):
    '''
    This module output 4 values
        1. bx = x axis pixel coordinate top left corner
        2. by = y axis pixel coordinate top left corner
        3. log(bh) = height of bounding box
        4. log(bw) = width of bounding box
        
    This is a linear classifier
    :param x:
    :return:
    '''
    sh_in = x.get_shape().as_list()[-1]

    # Here 4*len(anchor_ratio) = 8, where 4 is the count of bounding box output
    x = ops.conv_layer(x, k_shape=[1, 1, sh_in, 4 * anchor_per_location], stride=anchor_stride, padding='VALID', scope_name='rpn_bbox_pred')
    logging.info('RPN - Conv Bbox: %s', str(x.get_shape().as_list()))

    # The shape of rpn_bbox = [None, None, 4] =  Which says for each image for each pixel position of a feature map the output of box is 4 -> center_x, center_y, width and height. Since we do it in pixel basis, we would end up having many many bounding boxes overlapping and hence we use non-max suppression to overcome this situation.
    rpn_bbox = tf.reshape(x, [tf.shape(x)[0], -1, 4])
    logging.info('rpn_bbox: %s', rpn_bbox.get_shape().as_list())
    
    return rpn_bbox
    



def rpn_graph(depth):
    xrpn = tf.placeholder(dtype=tf.float32, shape=[None, None, None, depth],
                          name='rpn_feature_map_inp')

    shared = ops.conv_layer(xrpn, k_shape=[3, 3, xrpn.get_shape().as_list()[-1], 512], stride=conf.RPN_ANCHOR_STRIDES,
                            padding='SAME', scope_name='rpn_conv_shared')
    shared = ops.activation(shared, 'relu', scope_name='rpn_relu_shared')
    logging.info('RPN - Shared_conv: %s', str(shared.get_shape().as_list()))

    ## Classification Output: Binary classification, # Get the pixel wise Classification
    rpn_class_logits, rpn_probs = get_pixel_fb_classification(
            shared, conf.RPN_ANCHOR_STRIDES,  len(conf.RPN_ANCHOR_RATIOS))
    
    ## Bounding Box Output: Get the coordinates , height and width of bounding box
    rpn_bbox = get_bounding_box(
            shared, conf.RPN_ANCHOR_STRIDES, len(conf.RPN_ANCHOR_RATIOS))
    
    return dict(
            xrpn=xrpn,
            rpn_class_logits=rpn_class_logits,
            rpn_probs=rpn_probs,
            rpn_bbox=rpn_bbox
    )
