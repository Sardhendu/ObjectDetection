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

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

class RPN():
    def __init__(self, depth):
        self.rpn_anchor_stride = conf.RPN_ANCHOR_STRIDE
        self.rpn_anchor_ratios = conf.RPN_ANCHOR_RATIOS
        self.xrpn = tf.placeholder(dtype=tf.float32, shape=[None, None, None, depth],
                              name='rpn_feature_map_inp')
        
        self.build()
    
    def build(self):
        shared = ops.conv_layer(self.xrpn, k_shape=[3, 3, self.xrpn.get_shape().as_list()[-1], 512],
                                     stride=conf.RPN_ANCHOR_STRIDE,
                                     padding='SAME', scope_name='rpn_conv_shared')
        shared = ops.activation(shared, 'relu', scope_name='rpn_relu_shared')
        logging.info('RPN - Shared_conv: %s', str(shared.get_shape().as_list()))
        
        ## Classification Output: Binary classification, # Get the pixel wise Classification
        self.get_pixel_fb_classification(shared, self.rpn_anchor_stride, len(self.rpn_anchor_ratios))

        ## Bounding Box Output: Get the coordinates , height and width of bounding box
        self.get_bounding_box(shared, self.rpn_anchor_stride, len(self.rpn_anchor_ratios))
            
    def get_pixel_fb_classification(self, x, anchor_stride, anchor_per_location):
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
        self.rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
        # self.rpn_class_logits = tf.reshape(x, [x.get_shape().as_list()[0], -1, 2])
        logging.info('rpn_class_logits: %s', self.rpn_class_logits.get_shape().as_list())


        # Do a softmax classificaion to get output probabilities
        self.rpn_class_probs = tf.nn.softmax(self.rpn_class_logits, name='rpn_class_xxx')
        logging.info('rpn_class_probs: %s', self.rpn_class_probs.get_shape().as_list())

        print('(RPN) Class Logits (shape) ', self.rpn_class_logits.shape)
        print('(RPN) Class Probs (shape) ', self.rpn_class_probs.shape)

    def get_bounding_box(self, x, anchor_stride, anchor_per_location):
        '''
        ALL ABOUT THIS MODULE
        
        Input:
        anchor_stride: controls the number of anchors,
            for instance: if stride = 1, feature_map = 32x32, num_anchors = 9
                          then number of anchors = 32 x 32 x 9
                          if stride = 2, feature_map = 32x32, num_anchors = 9
                          then number of anchors = (32 x 32 x 9)/2
        anchor_per_location: How many anchors to build per location
                          
        
        Outputs:
        This module generates4 values
        self.rpn_bbox = [batch_size, h, w, (dy, dx, log(dh), log(dw))]
            1. dy = center y pixel
            2. dx = center x pixel
            3. log(dh) = height of bounding box
            4. log(dw) = width of bounding box
            
        This is a linear classifier
        :param x:
        :return:
        '''
        sh_in = x.get_shape().as_list()[-1]
    
        # Here 4*len(anchor_ratio) = 8, where 4 is the count of bounding box output
        x = ops.conv_layer(x, k_shape=[1, 1, sh_in, 4 * anchor_per_location], stride=anchor_stride, padding='VALID', scope_name='rpn_bbox_pred')
        logging.info('RPN - Conv Bbox: %s', str(x.get_shape().as_list()))
    
        # The shape of rpn_bbox = [None, None, 4] =  Which says for each image for each pixel position of a feature map the output of box is 4 -> center_x, center_y, width and height. Since we do it in pixel basis, we would end up having many many bounding boxes overlapping and hence we use non-max suppression to overcome this situation.
        self.rpn_bbox = tf.reshape(x, [tf.shape(x)[0], -1, 4])
        # self.rpn_bbox = tf.reshape(x, [x.get_shape().as_list()[0], -1, 4])
        logging.info('rpn_bbox: %s', self.rpn_bbox.get_shape().as_list())
        print('(RPN) Bbox (shape) ', self.rpn_bbox.shape)
        
    def get_rpn_class_logits(self):
        return self.rpn_class_logits
    
    def get_rpn_class_probs(self):
        return self.rpn_class_probs
    
    def get_rpn_bbox(self):
        return self.rpn_bbox
    
    def get_rpn_graph(self):
        return dict(
                xrpn=self.xrpn,
                rpn_class_logits=self.rpn_class_logits,
                rpn_class_probs=self.rpn_class_probs,
                rpn_bbox=self.rpn_bbox
        )
    
    
    
# RPN(depth=256)

