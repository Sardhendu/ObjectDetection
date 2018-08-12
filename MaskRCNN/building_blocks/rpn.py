'''

As discussed in the notes section, RPN will have two outputs,
    1) Classify a pixel point as foreground or background
    2) Classify the anchor and give a bounding box for it. For example if we are trying to evaluate 9 anchors,
    then the output at each pixel point would be 9x4 = 36
'''


import logging
import numpy as np
import tensorflow as tf
from MaskRCNN.building_blocks import ops

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


class RPN():
    def __init__(self, conf, depth, feature_map=None):
        self.rpn_anchor_stride = conf.RPN_ANCHOR_STRIDE
        self.rpn_anchor_ratios = conf.RPN_ANCHOR_RATIOS
        
        if feature_map is not None:
            self.xrpn = feature_map
        else:
            self.xrpn = tf.placeholder(dtype=tf.float32, shape=[None, None, None, depth],
                              name='rpn_feature_map_inp')
        

        self.build_keras()
        
    def build_keras(self):
        # rpn = build_rpn_model(self.rpn_anchor_stride,
        #                       len(self.rpn_anchor_ratios), 256)
        # rpn_feature_maps = [P2, P3, P4, P5, P6]
        # layer_outputs = []  # list of lists
        # for p in range(0,5):
        #     layer_outputs.append(rpn_graph(self.xrpn[p], self.rpn_anchor_stride, len(self.rpn_anchor_ratios)))

        logging.info('rpn_graph .....')
        # TODO: check if stride of 2 causes alignment issues if the featuremap
        #       is not even.
        # Shared convolutional base of the RPN
        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                           strides=self.rpn_anchor_stride,
                           name='rpn_conv_shared')(self.xrpn)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = KL.Conv2D(2 * len(self.rpn_anchor_ratios), (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        self.rpn_class_logits = KL.Lambda(
                lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

        # Softmax on last dimension of BG/FG.
        self.rpn_class_probs = KL.Activation(
                "softmax", name="rpn_class_xxx")(self.rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        x = KL.Conv2D(len(self.rpn_anchor_ratios) * 4, (1, 1), padding="valid",
                      activation='linear', name='rpn_bbox_pred')(shared)

        # Reshape to [batch, anchors, 4]
        self.rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

        
        # self.rpn_class_logits, self.rpn_class_probs, self.rpn_bbbox = outputs

    def build(self):
        shared = ops.conv_layer(self.xrpn,
                                k_shape=[3, 3, self.xrpn.get_shape().as_list()[-1], 512],
                                stride=self.rpn_anchor_stride,
                                padding='SAME', scope_name='rpn_conv_shared',
                                trainable=True)
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
        x = ops.conv_layer(x, k_shape=[1, 1, sh_in, 2 * anchor_per_location], stride=anchor_stride, padding='VALID', scope_name='rpn_class_raw', trainable=True)
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
        This module generates 4 values
        self.rpn_bbox = [batch_size, num_anchors, (dy, dx, log(dh), log(dw))]
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
        x = ops.conv_layer(x, k_shape=[1, 1, sh_in, 4 * anchor_per_location], stride=anchor_stride, padding='VALID', scope_name='rpn_bbox_pred', trainable=True)
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



def debug():
    
    from MaskRCNN.config import config as conf
    fpn_p2 = tf.constant(np.random.random((1, 32, 32, 256)), dtype=tf.float32)
    fpn_p3 = tf.constant(np.random.random((1, 16, 16, 256)), dtype=tf.float32)
    fpn_p4 = tf.constant(np.random.random((1, 8, 8, 256)), dtype=tf.float32)
    fpn_p5 = tf.constant(np.random.random((1, 4, 4, 256)), dtype=tf.float32)

    rpn_class_probs = []
    rpn_bbox = []
    for fmap in [fpn_p2,fpn_p3,fpn_p4,fpn_p5]:
        rpn_obj = RPN(conf, depth=256, feature_map=fmap)
        rpn_class_probs.append(rpn_obj.get_rpn_class_probs())
        rpn_bbox.append(rpn_obj.get_rpn_bbox())

    rpn_class_probs = tf.stack(rpn_class_probs, axis=0)
    rpn_bbox = tf.stack(rpn_bbox, axis=0)
    print(rpn_class_probs.shape)
    print(rpn_bbox.shape)


# debug()







######################################################################################################
####################################################################
################################## Keras:

import keras.backend as K
import keras.layers as KL

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    logging.info('rpn_graph .....')
    # TODO: check if stride of 2 causes alignment issues if the featuremap
    #       is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location, depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]

# def build_rpn_model(anchor_stride, anchors_per_location, depth):
#     """Builds a Keras model of the Region Proposal Network.
#     It wraps the RPN graph so it can be used multiple times with shared
#     weights.
#
#     anchors_per_location: number of anchors per pixel in the feature map
#     anchor_stride: Controls the density of anchors. Typically 1 (anchors for
#                    every pixel in the feature map), or 2 (every other pixel).
#     depth: Depth of the backbone feature map.
#
#     Returns a Keras Model object. The model outputs, when called, are:
#     rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
#     rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
#     rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
#                 applied to anchors.
#     """
#     logging.info('build_rpn_model .....')
#     input_feature_map = KL.Input(shape=[None, None, depth],
#                                  name="input_rpn_feature_map")
#     outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
#     return outputs#KM.Model([input_feature_map], outputs, name="rpn_model")



