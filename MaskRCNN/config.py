



class config(object):

    IMAGE_SHAPE = [1024, 1024, 3]
    BATCH_NORM_DECAY = 0.9
    RPN_ANCHOR_STRIDES = 1
    
    # 0.5 indicates the horizontal axis of the anchor is twice the vertical axis
    # 1 indicates the horizontal axis of the anchor is equal to the vertical axis
    # 2 indicates the horizontal axis of the anchor is half the vertical axis
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    

