'''
Till this point We have already performed the FPN(feature pyramid network) and "Region Proposal Network". As an output from the RPN net we have:
    1. rpn_class_logits: [batch_size, pixel_position * num_anchor, 2]:
        This gives a binary outcome, if an anchor at a pixel for a image is foreground or background
    2. rpn_class_logits: [batch_size, pixel_position * num_anchor, 2]:
        This are just sigmoid outcomes of the Logits
    3. rpn_bbox: [batch_size, pixel_position * num_anchors, 4]:
        This outputs continuous values that outputs the bounding box of the anchors
        
Problem: For 1 pixel position we can have multiple anchors that can qualify as a bounding box for an object. Therefore in this module we take care of overlaps and select only the bounding box that has high IOU. This is also implemented using non-max supression.
        
'''


import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def proposals(conf, inputs):
    '''
    :param config:
    :param inputs:
        inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, (y1, x1, y2, x2)] anchors in normalized coordinates
    :return:
    '''
    
    # We would like to only capture the foreground class probabilities
    foreground_probs = inputs[0][:,:,1]
    logging.info('Foreground_probs shape: %s', str(foreground_probs.get_shape().as_list()))
    
    # Box deltas = [batch, num_rois, 4]
    box_delta = inputs[1] * np.reshape(conf.RPN_BBOX_STD_DEV, [1, 1, 4])
    logging.info('Foreground_probs shape: %s', str(foreground_probs.get_shape().as_list()))
    
    # Get the anchors [None, 2]
    anchors = inputs[2]
    logging.info('Foreground_probs shape: %s', str(tf.shape(anchors)))
    
    # Searching through lots of anchors can be time consuming. So we would select at most 6000 of them for further processing
    max_anc_before_nms = tf.minimum(6000, tf.shape(anchors)[1])
    ix = tf.nn.top_k(foreground_probs, max_anc_before_nms, sorted=True,
                         name="top_anchors").indices
    
    
def debugg():
    from MaskRCNN.config import config as conf

    rpn_probs = tf.placeholder(dtype=tf.float32,
                                   shape=[None, None, 2],
                                   name="rpn_prob")

    rpn_box = tf.placeholder(dtype=tf.float32,
                                   shape=[None, None, 4],
                                   name="rpn_box")
    
    
    input_anchors = tf.placeholder(dtype=tf.float32,
                                   shape=[None, None, 4],
                                   name="input_anchors")
    proposals(conf, inputs=[rpn_probs, rpn_box, input_anchors])


debugg()