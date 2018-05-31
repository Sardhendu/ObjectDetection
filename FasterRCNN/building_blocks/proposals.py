
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

from FasterRCNN.config import config as conf

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

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


def get_anchors():
    '''Shaoqing's matlab implementation: ZF net anchors
    
    Just 9 anchors of different shapes to generalize different types of objects
    the format of the anchor is:
        
        [c_y, c_x, h, w]
    '''
    return np.array([[ -84.,  -40.,   99.,   55.],
                     [-176.,  -88.,  191.,  103.],
                     [-360., -184.,  375.,  199.],
                     [ -56.,  -56.,   71.,   71.],
                     [-120., -120.,  135.,  135.],
                     [-248., -248.,  263.,  263.],
                     [ -36.,  -80.,   51.,   95.],
                     [ -80., -168.,   95.,  183.],
                     [-168., -344.,  183.,  359.]])


class Proposals():
    def __init__(self, mode, rpn_box_class_prob, rpn_bbox):
        self.rpn_box_class_prob = rpn_box_class_prob
        self.rpn_bbox = rpn_bbox
        if mode == 'train':
            self.PRE_NMS_TOP_N = 12000
            self.POST_NMS_TOP_N = 2000
            self.NMS_THRESHOLD = 0.7
            self.MIN_SIZE = 16
        else:
            self.PRE_NMS_TOP_N = 6000
            self.POST_NMS_TOP_N = 300
            self.NMS_THRESHOLD = 0.7
            self.MIN_SIZE = 16
            
        self.build()
        
    def build(self):
        '''
        Understand whats happening
        
        Theory:
        
        Stage_1 =
        
        Stage_2 = Get only the foreground probabilities
        We have 18 (9*2) anchor probabilities, we take that the first 9 values corresponds to Foreground
        probabilities and the Last 9 values corresponds to Background probabilities. We consider only the
        foreground probabilities
        
        Stage_3 = Get pixels position to specify anchors:
        In practise we have to interpolate the anchors on the original image that is 224x224 and the feature_map size is 14x14. Which says that the center pixel position in the original image would be every point at a stride of 16 (224/14 = 16). So we create a mesh grid of each pixel position that we consider to be the center to place anchors. Total pixel position = 14x14 = 196 (feature map shape). So to concrete it, we would require a 196x4 shape matrix where 196 is the pixel position and 4 is the (c_y, c_x, h, w) anchor bbox.
        
            Generate a matrix just like this
             shifts  =  [[  0   0   0   0]
                         [ 16   0  16   0]
                         [ 32   0  32   0]
                         [ 48   0  48   0]
                         [ 64   0  64   0]
                         [ 80   0  80   0]
                         [ 96   0  96   0]
                         [112   0 112   0]
                         [128   0 128   0]
                         [144   0 144   0]
                         [160   0 160   0]
                         [176   0 176   0]
                         [192   0 192   0]
                         [208   0 208   0]
                         [  0  16   0  16]
                         [ 16  16  16  16]
                         [ 32  16  32  16]
                         [ 48  16  48  16]
        
        Stage 4:
        From the previous step we get 196*4 where 196 is the shifts (of number of center pixel coordinates). Also
        from stage 1 we have 9 different anchors bbox. In total we would have 196*9 = 1764 anchors bbox for all the
        pixel position in the original image. Note the 9 anchors we have depicts the pixel position (0,
        0) of the image 224x224, in order to capture different pixels position we have to add 16 to the c_x while
        shifting in x direction and add 16 to c_y while shifting in y direction: Basically we add the matrix
        generated from stage_3. Also, as we shift we would like to try all different heights and widths of anchors.
        
        anchors_ = 1764x4   [[ -84.  -40.   99.   55.]  # [-84,  -40.,   99,   55.]
                             [-176.  -88.  191.  103.]  # [-176.  -88.  191.  103.]
                             [-360. -184.  375.  199.]  # [-360. -184.  375.  199.]
                             [ -56.  -56.   71.   71.]  # [ -56.  -56.   71.   71.]
                             [-120. -120.  135.  135.]  # [-120. -120.  135.  135.] + [ 0 0 0 0] --> (shifts[0])
                             [-248. -248.  263.  263.]  # [-248. -248.  263.  263.]
                             [ -36.  -80.   51.   95.]  # [ -36.  -80.   51.   95.]
                             [ -80. -168.   95.  183.]  # [ -80. -168.   95.  183.]
                             [-168. -344.  183.  359.]  # [-168. -344.  183.  359.]
                             
                            [[ -68.  -40.  115.   55.]  # [-84,  -40.,   99,   55.]
                             [-160.  -88.  207.  103.]  # [-176.  -88.  191.  103.]
                             [-344. -184.  391.  199.]  # [-360. -184.  375.  199.]
                             [ -40.  -56.   87.   71.]  # [ -56.  -56.   71.   71.]
                             [-104. -120.  151.  135.]  # [-120. -120.  135.  135.] + [ 16 0 16 0] --> (shifts[1])
                             [-232. -248.  279.  263.]  # [-248. -248.  263.  263.]
                             [ -20.  -80.   67.   95.]  # [ -36.  -80.   51.   95.]
                             [ -64. -168.  111.  183.]  # [ -80. -168.   95.  183.]
                             [-152. -344.  199.  359.]] # [-168. -344.  183.  359.]
        '''
        
        # Stage1:
        anchors_ = get_anchors()

        num_anchors = anchors_.shape[0]  # Should be 9
        print (self.rpn_box_class_prob.shape, self.rpn_bbox.shape)
        rpn_bbox_class_prob = np.transpose(self.rpn_box_class_prob, [0, 3, 1, 2])  # [1, 9*2, height, width ]
        rpn_bbox = np.transpose(self.rpn_bbox, [0, 3, 1, 2])  # [1, 9*4, height, width ]
        
        logging.info('rpn_bbox_cls_prob reshaped: %s', str(rpn_bbox_class_prob.shape))
        logging.info('rpn_bbox reshaped: %s', str(rpn_bbox.shape))
        
        # Stage 2
        fg_prob = rpn_bbox_class_prob[:,:num_anchors, :,:]
        logging.info('fg_prob shape %s', str(fg_prob.shape))

        # Stage 3:
        height, width = fg_prob.shape[-2:]
        shift_x = np.arange(0, width) * conf.RPN_FEATURE_STRIDE  # [0,16,32,48,64,80,96,112,128,144,160,176,192,208]
        shift_y = np.arange(0, height) * conf.RPN_FEATURE_STRIDE # [0,16,32,48,64,80,96,112,128,144,160,176,192,208]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()))
        shifts = shifts.transpose()
        num_shifts = shifts[0]
        
        print('')
        print(shift_x)
        print(shift_y)
        print ('')
        print(shifts.shape, shifts)


        # Stage 4:
        anchors = anchors_.reshape((1, num_anchors, 4)) + shifts.reshape(1, num_shifts, 4).transpose((1, 0, 2))
        print('anchors ', anchors.shape, anchors)
        anchors = anchors.reshape((num_shifts * num_anchors), 4)
        print('')
        # print ('anchors ', anchors.shape, anchors[0:9,:])
        # print('anchors ', anchors.shape, anchors[9:18, :])
        # print('anchors ', anchors.shape, anchors[18:27, :])
        # print('anchors ', anchors.shape, anchors[27:36, :])
        # print('anchors ', anchors.shape, anchors[36:45, :])
        # print('anchors ', anchors.shape, anchors[45:54, :])
        # print('anchors ', anchors.shape, anchors[54:63, :])
        
        
        # FILTERING CONDITIONS
        
        
        
        
def debugg():
    # PROPOSALS
    rpn_box_class_prob = np.random.random((1, 14, 14, 18))
    rpn_bbox = np.random.random((1, 14, 14, 36))
    Proposals('test', rpn_box_class_prob=rpn_box_class_prob, rpn_bbox=rpn_bbox)
    
debugg()
