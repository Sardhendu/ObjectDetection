

import numpy as np

class config(object):
    
    # RPN
    RPN_KERNEL_SIZE = 3
    RPN_OUT_CHANNEL = 512
    ANCHOR_PER_LOCATION = 9  # 9 anchors and each classified as object or non-object
    
    # PROPOSALS
    BASE_ANCHOR = [1, 1, 15, 15]  # [1,1,16,16] - 1
    ANCHOR_RATIO = [0.5, 1, 2]
    ANCHOR_SCALES = 2* np.arange(3,6)