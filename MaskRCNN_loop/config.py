

import numpy as np

class config(object):

    IMAGE_SHAPE = [1024, 1024, 3]
    NUM_CLASSES = 1
    
    # Image Pre-processing params
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_SCALE = 0
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    
    
    BATCH_NORM_DECAY = 0.9

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    RESNET_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_STRIDE = 1
    
    # 0.5 indicates the horizontal axis of the anchor is twice the vertical axis
    # 1 indicates the horizontal axis of the anchor is equal to the vertical axis
    # 2 indicates the horizontal axis of the anchor is half the vertical axis
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STDDEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of anchors before NMS
    PRE_NMS_ROIS_INFERENCE = 6000
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    
    # Non-max suppression threshold to filter RPN proposals.
    # This can be increase during training to generate more proposals.
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_MIN_THRESHOLD = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_POST_NMS_INSTANCES = 100
    
    

    
