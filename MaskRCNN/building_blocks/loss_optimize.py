

import tensorflow as tf


def rpn_class_loss(rpn_target_class, rpn_class_logits):
    '''
    
    :param rpn_target_class:  target class [batch_size, num_anchors]
    :param rpn_class_probs:   predicted class [batch_size, num_anchors, (fg_probs, bg_probs)]
    
    :return: loss (scalar)
    '''
    
    tf.argmax(rpn_class_probs)