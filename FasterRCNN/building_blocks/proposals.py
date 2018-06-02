
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


def non_max_suppression_fast(boxes, scores, overlapThresh, max_output_size):
    '''
    FEW IMPORTANT NOTES:
    
    1. This module takes a vector approach where we have have the inner for loop that loop through all other boxes
    given the running box
    2. Note inorder to compute the non-max suppression in a fast way, we don't actually do Intersection over union.
    We do an approximate overlap. If we do intersection over union then we may have to loop through all other threshold.
    
    '''
    
    # print('')
    # print('NMS 2 ............')
    # print ('')
    scores = scores.ravel()
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    # initialize the list of keeped indexes
    keep = []
    
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # print ('area ', area)
    idxs = scores.argsort()#[::-1]

    # print('idx ', idxs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # print('idx .. ', idxs)
        # grab the last index in the indexes list and add the
        # index value to the list of keeped indexes
        last = len(idxs) - 1
        i = idxs[last]
        keep.append(i)
        
        # Here we basically find the lower left coordinate and upper right coordinate of the intersecting space of the box with highest score (x1[last]) with all other boxes.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        # print('xx1', x1[i], x1[idxs[:last]], xx1)
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        # print('yy1', y1[i], y1[idxs[:last]], yy1)
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        # print('xx2', x2[i], x2[idxs[:last]], xx2)
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # print('yy2', y2[i], y2[idxs[:last]], yy2)
        
        # To handle the case where xx2 < xx1 or yy2 < yy1. Because when we say lower left and upper right then xx1<xx2 and yy1<yy2
        w = np.maximum(0, np.maximum(xx2 - xx1, 1))
        h = np.maximum(0, np.maximum(yy2 - yy1, 1))
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        # print ('Area : ', w, h, w*h)
        
        # compute the ratio of overlap =  intersection_area / area_of_rectangle_in_while_loop.
        overlap = (w * h) / area[idxs[:last]]
        # print('Overlap: ', overlap)
        
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    
    # return only the bounding boxes that were keeped using the
    # integer data type
    if max_output_size < len(keep):
        return boxes[keep[0:max_output_size],:]
    else:
        return boxes[keep,:]

def non_max_suppression(boxes, scores, iou_thresh, max_output_size):
    # print('NMS 1 ............')
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = scores.ravel()
    
    ndets = boxes.shape[0]
    order = scores.argsort()[::-1]
    suppressed = np.zeros((ndets), dtype=np.int)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= iou_thresh:
                suppressed[j] = 1

    if max_output_size < len(keep):
        return boxes[keep[0:max_output_size],:]
    else:
        return boxes[keep,:]
    
    
def non_max_suppression_tf(boxes, score, iou_thresh, max_output_size):
    nms_indices = tf.image.non_max_suppression(boxes,
                                               score,
                                       max_output_size=max_output_size,
                                       iou_threshold=iou_thresh,
                                       name='activeBox_indice')
    return tf.gather(boxes, nms_indices)

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

def corner_pixels_to_center(anchors, bbox_pred):
    '''
    See Whats actually happening

    Anchor we use in the (get_anchor) method has a form: [bottom_left_x, bottom_left_y, upper_right_x, upper_right_y],
    but in the Faster-RCNN paper the author designs a formula that computes a proposals using the [center_x,
    center_y, height, weight]. In this function we do the below
    
        1. First convert the corner coordinates to center_x, center_y, height and width
        2. Implement the equation 2 in the Faster RCNN paper
            t_x = (x_pred - x_anchor)/w_anchor
            t_y = (y_pred - y_anchor)/h_anchor
            
            t_w = log(w_pred/w_anchor)
            t_h = log(h_pred/h_anchor)
    '''
    
    # Ground truth boxes (Convert to center coordinates, width, and height)
    bbox_width = bbox_pred[:,2] - bbox_pred[:,0] + 1
    bbox_height = bbox_pred[:, 3] - bbox_pred[:, 1] + 1
    bbox_cx = bbox_pred[:,0] + bbox_width / 2
    bbox_cy = bbox_pred[:,1] + bbox_height / 2

    # Anchor boxes (Convert to center coordinates, width, and height)
    anchor_width = anchors[:, 2] - anchors[:, 0] + 1
    anchor_height = anchors[:, 3] - anchors[:, 1] + 1
    anchor_cx = anchors[:, 0] + anchor_width / 2
    anchor_cy = anchors[:, 1] + anchor_height / 2

    # APPLY FORMULAS
    t_x = (bbox_cx - anchor_cx) / anchor_width
    t_y = (bbox_cy - anchor_cy) / anchor_height

    # Perform Log transform
    t_w = np.log(bbox_width / anchor_width)
    t_h = np.log(bbox_height / anchor_height)
    
    targets = np.vstack((t_x, t_y, t_w, t_h))
    return targets.transpose()
    

def corner_pixels_to_center_inv(anchor_boxes, pred_box_deltas):
    '''
    Understand whats happening
    
    :param anchor_boxes:
    :param pred_box_deltas:
    :return:
    
    1. The pred_box_deltas are basically regression outputs that are sqashed between 0 and 1.
        Suppose  pred_box_deltas[0] = [ 0.2  0.8  0.2  0.8], this means that for the box an
        given below then the inner point defines the corner in range of (0,1)
        
                    ______________1,1
                    |  (.8,.8) . |
                    |            |
                    |            |
                    | . (.2,.2)  |
                0,0 --------------
    
    2. However, the anchor boxes represents values in integers which when transformed with the corner box prediction
    would represent the corner coordinates in the origin image
    
        
        The question we have to ask ourselves is that, given we are at a pixel position in the original image with
        pred_box_deltas = [ 0.2  0.8  0.2  0.8], anchor_width = 184, anchor_height = 96, anchor_cx = 8, anchor_cy = 8
        then, what is the center_x, center_y in the original image.
        
        
                              184
                    <--- anchor_width ---->
                    ______________________ (0.8,0.8)   ___
                    |                     |             |
                    |                     |             |
                    |                     |             |
                    |          .          |       anchor height = 96
                    | (pred_cx, pred_cy)  |             |
                    |                     |             |
                    |                     |             |
        (0.2, 0.2)  -----------------------            ---
        
    '''
    if anchor_boxes.shape[0] == 0:
        return np.zeros((0, pred_box_deltas[1]), dtype=pred_box_deltas.dtype)

    anchor_boxes = anchor_boxes.astype(pred_box_deltas.dtype, copy=False)

    # Anchor boxes (Convert to center coordinates, width, and height)
    anchor_width = anchor_boxes[:, 2] - anchor_boxes[:, 0] + 1
    anchor_height = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1
    anchor_cx = anchor_boxes[:, 0] + anchor_width / 2
    anchor_cy = anchor_boxes[:, 1] + anchor_height / 2

    dx = pred_box_deltas[:, 0].reshape(-1,1)    # In 0-1 range
    dy = pred_box_deltas[:, 1].reshape(-1,1)    # In 0-1 range
    dw = pred_box_deltas[:, 2].reshape(-1,1)    # In logarithmic scale
    dh = pred_box_deltas[:, 3].reshape(-1,1)    # In logarithmic scale
    
    # Prediction Boxes = anchor_boxes + pred_box_deltas
    pred_cx = dx * anchor_width.reshape(-1,1) + anchor_cx.reshape(-1,1)
    pred_cy = dy * anchor_height.reshape(-1,1) + anchor_cy.reshape(-1,1)
    # Transform weights and heights from logarithmic scale to normal scale
    pred_w = np.exp(dw) * anchor_width.reshape(-1,1)
    pred_h = np.exp(dh) * anchor_height.reshape(-1,1)
    
    # Convert the prection boxes back to corner points
    pred_boxes = np.zeros(pred_box_deltas.shape, dtype=pred_box_deltas.dtype)#.reshape(-1,4)
    # print ((pred_cx - pred_w / 2).shape)
    pred_boxes[:, 0::4] = pred_cx - pred_w / 2  # bottom_left_x
    pred_boxes[:, 1::4] = pred_cy - pred_h / 2  # bottom_left_y
    pred_boxes[:, 2::4] = pred_cx + pred_w / 2  # upper_right_x
    pred_boxes[:, 3::4] = pred_cy + pred_h / 2  # upper_right_y
    
    return pred_boxes



class FilterBoxes():
    def __init__(self, image_shape, min_box_hw, pre_nms_top_n, post_nms_top_n, iou_thresh, boxes, scores):
        '''
        OPERATION PERFORMED:
        
        1. Clip exceeding box corners to image shape
        2. Filter in boxes are smaller than 16x16
        2. Select top n boxes based on scores (rpn_class_prob)
        3. Filter boxes with non-max-suppression
        '''
        self.image_shape = image_shape
        self.min_box_hw = min_box_hw
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.iou_thresh = iou_thresh
        
        self.boxes = boxes
        self.scores = scores
        
    def clip_boxes(self):
        self.boxes[:, 0] = np.maximum(np.minimum(self.boxes[:, 0], self.image_shape[1] - 1), 0)
        self.boxes[:, 1] = np.maximum(np.minimum(self.boxes[:, 1], self.image_shape[0] - 1), 0)
        self.boxes[:, 2] = np.maximum(np.minimum(self.boxes[:, 2], self.image_shape[1] - 1), 0)
        self.boxes[:, 3] = np.maximum(np.minimum(self.boxes[:, 3], self.image_shape[0] - 1), 0)
        print('clip_boxes self.boxes ', self.boxes.shape)
        
    def filter_min_size(self):
        self.keep_idx = np.where(
                (self.boxes[:, 2] - self.boxes[:,0 ] + 1  >= self.min_box_hw) &
                (self.boxes[:, 3] - self.boxes[:, 1] + 1 >= self.min_box_hw)
        )[0]

        self.boxes = self.boxes[self.keep_idx, :]
        self.scores = self.scores[self.keep_idx, :]
        print('filter_min_size self.boxes ', self.boxes.shape)
    
    def non_max_suppression(self):
        ordered_idx_desc = self.scores.argsort()[::-1]
        if self.pre_nms_top_n < len(ordered_idx_desc):
            ordered_idx_desc = ordered_idx_desc[0:self.pre_nms_top_n]
        ordered_idx_desc = ordered_idx_desc.ravel()
        
        self.scores = self.scores[ordered_idx_desc, :]
        self.boxes = self.boxes[ordered_idx_desc, :]

        self.boxes = non_max_suppression(self.boxes, self.scores, self.iou_thresh, self.post_nms_top_n)
        print ('non_max_suppression self.boxes ', self.boxes.shape)
        
    def get_filtered_boxes(self):
        self.clip_boxes()
        self.filter_min_size()
        self.non_max_suppression()
        
        return self.boxes
    
        

class Proposals():
    def __init__(self, mode, rpn_box_class_prob, rpn_bbox):
        self.rpn_box_class_prob = rpn_box_class_prob
        self.rpn_bbox = rpn_bbox
        
        if mode == 'train':
            self.PRE_NMS_TOP_N = 12000
            self.POST_NMS_TOP_N = 2000
            self.NMS_THRESHOLD = 0.7
            self.MIN_BOX_HW = 16        # Min height and width of a box
        else:
            self.PRE_NMS_TOP_N = 6000
            self.POST_NMS_TOP_N = 300
            self.NMS_THRESHOLD = 0.7
            self.MIN_BOX_HW = 16          # Min height and width of a box
        
        self.IMAGE_SHAPE = [224,224,3]
        self.build()
    
    def build(self):
        '''
        Understand whats happening

        Theory:

        Stage_1 = Get only the foreground probabilities
        We have 18 (9*2) anchor probabilities, we take that the first 9 values corresponds to Foreground
        probabilities and the Last 9 values corresponds to Background probabilities. We consider only the
        foreground probabilities. We also reshape the rpn_class_prob and rpn_bbox to a flattened dimension

        Stage_2 = Get pixels position to specify anchors:
        In practise we have to interpolate the anchors on the original image that is 224x224 and the
        feature_map size is 14x14. Which says that the center pixel position in the original image would be
        every point at a stride of 16 (224/14 = 16). So we create a mesh grid of each pixel position that we
        consider to be the center to place anchors. Total pixel position = 14x14 = 196 (feature map shape).
        So to concrete it, we would require a 196x4 shape matrix where 196 is the pixel position and 4 is the
        (bottom_left_x, bottom_left_y, upper_right_x, upper_right_y) anchor bbox.

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

        Stage 3:
        From the previous step we get 196*4 where 196 is the shifts (of number of center pixel coordinates). Also
        from stage 1 we have 9 different anchors bbox. In total we would have 196*9 = 1764 anchors bbox for all the
        pixel position in the original image. Note the 9 anchors we have depicts different shapes. The format anchors are
        chosen is [lower_left_x, lower_left_y, upper_right_x, upper_right_y]. So we have a anchor [ -84.  -40.   99.   55.]
        then the
            Height of the anchor box is 55 - (-40) = 95
            Width of anchor box is in 99 - (-84) = 183
            
        We say at image[0,0] we have one anchor [ -84.  -40.   99.   55.] whose height is 95 and width is 183. In-orrder
        to capture different pixels position we have to add 16 to the corner position. Basically we add the matrix
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
                             [-104. -120.  151.  135.]  # [-120. -120.  135.  135.] + [ 16 0 16 0] --> (
                             shifts[1])
                             [-232. -248.  279.  263.]  # [-248. -248.  263.  263.]
                             [ -20.  -80.   67.   95.]  # [ -36.  -80.   51.   95.]
                             [ -64. -168.  111.  183.]  # [ -80. -168.   95.  183.]
                             [-152. -344.  199.  359.]] # [-168. -344.  183.  359.]
                             
        Stage 4: Up until now we have only anchors of different shape or (the 4 coordinate representing the corner
        value of the bounding box), now we need to transform these anchors as proposal in the image
        '''
        
        print ('')
        print ('................... build 2')
        logging.info('build')
        
        # Stage1:
        anchors_ = get_anchors()
        num_anchors = anchors_.shape[0]  # Should be 9
        self.rpn_box_class_prob = self.rpn_box_class_prob[:, :, :, :num_anchors]  # Get the foreground probs
        scores = self.rpn_box_class_prob.reshape((-1, 1))
        bbox_delta = self.rpn_bbox.reshape((-1, 4))
    
        # print ('scores: rpn_bbox_cls_prob reshaped:', str(scores.shape))
        # print ('bbox_delta : rpn_bbox reshaped: %s', str(bbox_delta.shape))
    
        # Stage 2
        height, width = self.rpn_box_class_prob.shape[1:3]
        shift_x = np.arange(0, width) * conf.RPN_FEATURE_STRIDE  # [0,16,32,48,64,80,96,112,128,144,160,176,192,208]
        shift_y = np.arange(0, height) * conf.RPN_FEATURE_STRIDE  # [0,16,32,48,64,80,96,112,128,144,160,176,192,208]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()))
        shifts = shifts.transpose()
        num_shifts = shifts.shape[0]

        # Stage 3:
        anchors_ = anchors_.reshape((1, num_anchors, 4)) + shifts.reshape(1, num_shifts, 4).transpose((1, 0, 2))
        anchors_ = anchors_.reshape((num_shifts * num_anchors), 4)
        print('')
        print('anchors ', anchors_.shape, anchors_)
        
        print ('')
        print('bbox_delta : rpn_bbox reshaped ', bbox_delta.shape, bbox_delta)
        
        # Stage 4:
        self.proposals = corner_pixels_to_center_inv(anchors_, bbox_delta)
        
        # Stage 5: Filter proposals
        self.proposals = FilterBoxes(self.IMAGE_SHAPE, self.MIN_BOX_HW, self.PRE_NMS_TOP_N, self.POST_NMS_TOP_N,
                                self.NMS_THRESHOLD, self.proposals, scores).get_filtered_boxes()
        
    def get_proposals(self):
        return self.proposals


def get_proposal_wrapper(mode, rpn_box_class_prob, rpn_bbox):
    return Proposals(mode, rpn_box_class_prob, rpn_bbox).get_proposals()

       
        
def debugg():
    # PROPOSALS
    rpn_box_class_prob = np.random.random((1, 14, 14, 18))
    rpn_bbox = np.random.random((1, 14, 14, 36))
    proposals = get_proposal_wrapper('test', rpn_box_class_prob=rpn_box_class_prob, rpn_bbox=rpn_bbox)
    print (proposals.shape)
    
    
    # Understand non-max-suppression
    # proposals = np.array([[1, 1, 5, 5], [4, 0, 7, 1], [2, 2, 4, 4], [10, 10, 20, 20], [0, 0, 4, 4]])
    # # scores = np.array([0.9, 0.8, 1, 0.7, 1]).reshape(-1, 1)
    # scores = np.array([1, 1, 1, 1, 1]).reshape(-1, 1)
    # keep = non_max_suppression(proposals, scores, 0.7, 3)
    # keep_2 = non_max_suppression_fast(proposals, scores, 0.7, 3)
    # print(keep)
    # print ('')
    # print( keep_2)
    
# debugg()
