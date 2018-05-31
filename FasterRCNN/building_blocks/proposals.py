
'''
UNDERSTAND WHY/HOW THE STUFF HAPPENS:

This step is called after RPN. RPN generates two items
    1) class probabilities of each anchors at eah pixel position
    2) the bounding box pertaining to each anchors

The problem:
    1) Some boxes when projected back to images could lie outside the boundary of images
    2) If we consider each anchor bbox where the class probability states an object. Then we can end up having many
    thousands of such proposals, Also these proposals may not be very accuracy and could only cover a small part of
    the object.
    
The Solution:
    In-order to get rid of such proposals, we need to perform
    1. Boxes that lie outside the boundary are to be clipped
    2. Removal of proposals that do not meet the specific threshold
    3. Perform Non-max suppression to get rid of overlapping proposals

IMPORTANT NOTE: Citing the paper: Faster RCNN:
    An anchor is centered at the sliding window in question and has a scale and aspect ratio yeilding k=9 anchors at
    each sliding position.
    
    Theory:
    When we run the VGG network we input a image of 224x224 and output a feature map of 14x14 (not including the
    fully connected layers). We can say that we downsampled the image by 16 (224/14). This can also mean that every
    pixel in the feature map corresponds to a 16x16 window in the original window. For training purposes, we have to
    find bounding boxes in the original image therefore, we have to interpolate or project the feature map anchors onto the original image, which would cover a much larger area. Additionally, there should be exact 9 (with 3 different
    ration and 3 different scale) of them.
'''

import logging

import numpy as np
import tensorflow as tf


# def get_anc_cx_cy_h_w(anchor):
#     '''
#     Get the center_x, center_y, height and width of an anchor
#     :return:
#     '''
#     width = anchor[2] - anchor[0] + 1
#     height = anchor[3] - anchor[1] + 1
#     center_x = anchor[0] + (width - 1)/2
#     center_y = anchor[1] + (height - 1) / 2
#     print (anchor)
#     return center_x, center_y, width, height
#
# def get_anc_given_cx_cy_h_w():
#     pass
#
#
# def generate_anchors():
#     anchors = np.array([1, 1, 16, 16]) - 1 # = [0,0,15,15], where 0,0 is coordinate and
#     # 15,15 is the width and height
#     cx, cy, w, h = get_anc_cx_cy_h_w(anchors)
#     print (cx, cy, w, h)
    
  
# generate_anchors()



anchors_ = np.array([[ -84.,  -40.,   99.,   55.],
                     [-176.,  -88.,  191.,  103.],
                     [-360., -184.,  375.,  199.],
                     [ -56.,  -56.,   71.,   71.],
                     [-120., -120.,  135.,  135.],
                     [-248., -248.,  263.,  263.],
                     [ -36.,  -80.,   51.,   95.],
                     [ -80., -168.,   95.,  183.],
                     [-168., -344.,  183.,  359.]])


class Proposals():
    def __init__(self):
    
