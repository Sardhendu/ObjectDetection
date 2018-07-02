

import tensorflow as tf


def rpn_class_loss(rpn_target_class, rpn_class_logits):
    '''
    :param rpn_target_class:  target class [batch_size, num_anchors, 1], here -1 = negative_class(iou<0.3), +1 = positive_class (iou>0.7), 0 = neutral
    :param rpn_class_probs:   predicted class [batch_size, num_anchors, (fg_probs, bg_probs)]
    
    The idea is to only gather data for -1 and +1 and avoid neutrals (0)
    :return: loss (scalar)
    
    '''
    
    # Convert rpn_target_class to [batch_size, num_anchors] with logits values
    rpn_target_class = tf.squeeze(rpn_target_class, axis=-1)
    
    # Select +1 class and -1 class, since 0 neutral doesnt contribute to loss
    indices = tf.where(tf.not_equal(rpn_target_class, 0))
    rpn_target_class = tf.gather_nd(rpn_target_class, indices)
    
    # Sigmoid cross entropy requires classes 1 and 0, we change -1 class to 0
    anchor_class = tf.cast(tf.equal(rpn_target_class, 1), tf.int32)
    