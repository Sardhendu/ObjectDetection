#
#
# import tensorflow as tf
# import numpy as np
# import keras.backend as K
# import keras.layers as KL
# from MaskRCNN.building_blocks.maskrcnn import BatchNorm
#
#
# class MaskLayer():
#     def __init__(self):
#         pass
#
#     def build(self):
#
#     def build_fpn_mask_graph(rois, feature_maps, image_meta,
#                              pool_size, num_classes, train_bn=True):
#         """Builds the computation graph of the mask head of Feature Pyramid Network.
#
#         rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
#               coordinates.
#         feature_maps: List of feature maps from diffent layers of the pyramid,
#                       [P2, P3, P4, P5]. Each has a different resolution.
#         image_meta: [batch, (meta data)] Image details. See compose_image_meta()
#         pool_size: The width of the square feature map generated from ROI Pooling.
#         num_classes: number of classes, which determines the depth of the results
#         train_bn: Boolean. Train or freeze Batch Norm layres
#
#         Returns: Masks [batch, roi_count, height, width, num_classes]
#         """
#
#         print('RUNNING build_fpn_mask_graph ......................')
#
#         # ROI Pooling
#         # Shape: [batch, boxes, pool_height, pool_width, channels]
#         x = PyramidROIAlign([pool_size, pool_size],
#                             name="roi_align_mask")([rois, image_meta] + feature_maps)
#
#         # Conv layers
#         x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
#                                name="mrcnn_mask_conv1")(x)
#         x = KL.TimeDistributed(BatchNorm(),
#                                name='mrcnn_mask_bn1')(x, training=train_bn)
#         x = KL.Activation('relu')(x)
#
#         x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
#                                name="mrcnn_mask_conv2")(x)
#         x = KL.TimeDistributed(BatchNorm(),
#                                name='mrcnn_mask_bn2')(x, training=train_bn)
#         x = KL.Activation('relu')(x)
#
#         x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
#                                name="mrcnn_mask_conv3")(x)
#         x = KL.TimeDistributed(BatchNorm(),
#                                name='mrcnn_mask_bn3')(x, training=train_bn)
#         x = KL.Activation('relu')(x)
#
#         x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
#                                name="mrcnn_mask_conv4")(x)
#         x = KL.TimeDistributed(BatchNorm(),
#                                name='mrcnn_mask_bn4')(x, training=train_bn)
#         x = KL.Activation('relu')(x)
#
#         x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
#                                name="mrcnn_mask_deconv")(x)
#         x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
#                                name="mrcnn_mask")(x)
#         return x