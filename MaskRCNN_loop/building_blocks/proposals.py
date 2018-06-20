'''
Till this point We have already performed the FPN(feature pyramid network) and "Region Proposal Network". As an
output from the RPN net we have:
    1. rpn_class_logits: [batch_size, pixel_position * num_anchor, 2]:
        This gives a binary outcome, if an anchor at a pixel for a image is foreground or background
    2. rpn_class_logits: [batch_size, pixel_position * num_anchor, 2]:
        This are just sigmoid outcomes of the Logits
    3. rpn_bbox: [batch_size, pixel_position * num_anchors, 4]:
        This outputs continuous values that outputs the bounding box of the anchors

Problem: For 1 pixel position we can have multiple anchors that can qualify as a bounding box for an object.
Therefore in this module we take care of overlaps and select only the bounding box that has high IOU. This is also
implemented using non-max supression.

'''

import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def apply_box_deltas_per_image(pre_nms_anchors, bbox_delta):
    '''
    Applying Box Deltas to Anchors

    pre_nms_anchors = [num_anchors, (y1, x1, y2, x2)]
    self.bbox_delta = [num_anchors, (d(c_y), d(c_x), log(h), log(w))]

                _____________ (x2, y2)
                |           |
                |           |
                |           |
                |           |
                |           |
        (x1,y1) -------------

    Since our predictions are normalized and are in the form of [d(c_y), d(c_x), log(h), log(w)],
    we first convert our anchors to the form of [center_y, center_x, h, w] and then apply box deltas (to
    normalize anchors that have un-normalized coordinate values). After this we convert the pre_nms_anchors back
    to the
    original shape of [num_images, num_anchors, (y1, x1,y2, x2)]

    :return:
    '''
    height = pre_nms_anchors[:, 2] - pre_nms_anchors[:, 0]
    width = pre_nms_anchors[:, 3] - pre_nms_anchors[:, 1]
    center_y = pre_nms_anchors[:, 0] + 0.5 * height
    center_x = pre_nms_anchors[:, 1] + 0.5 * width
    
    # Apply Box Delta (A)
    center_y += bbox_delta[:, 0] * height
    center_x += bbox_delta[:, 1] * width
    height *= tf.exp(bbox_delta[:, 2])
    width *= tf.exp(bbox_delta[:, 3])
    
    # Convert back to (y1, x1, y2, x2)
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    
    out = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return out


def clip_boxes_to_01_per_image(anchor_delta, window):
    """
    Clips Boxes within the range 0,1

    :param box_delta: The anchor per pixel position boxes for each batch with 4 pixel coordinates.
    :param window: THe min and max coordinates of window (We use this because our predictions should lie i 0,
    1 range)
    :return:

    The idea is pretty basic here:
        1. We split the coordinates.
        2. Check if they lie in the window range, if not make them lie
        3. Then concat them back to the original shape
    More over bring the box coordinate prediction to the range of [0,1] also helps us performing the next
    step i.e
    non-max suppression
    """
    
    # Window: [0,0,1,1] # 0,0 represents the top left corner and 1,1 represents the bottom right corner
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(anchor_delta, 4, axis=1)
    
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    
    clipped_anchors = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_anchors")
    clipped_anchors.set_shape((clipped_anchors.shape[0], 4))
    return clipped_anchors


def apply_box_deltas_per_batch(pre_nms_anchors, bbox_delta):
    '''
    Applying Box Deltas to Anchors

    pre_nms_anchors = [num_images, num_anchors, (y1, x1, y2, x2)]
    self.bbox_delta = [num_images, num_anchors, (d(c_y), d(c_x), log(h), log(w))]

                _____________ (x2, y2)
                |           |
                |           |
                |           |
                |           |
                |           |
        (x1,y1) -------------

    Since our predictions are normalized and are in the form of [d(c_y), d(c_x), log(h), log(w)],
    we first convert our anchors to the form of [center_y, center_x, h, w] and then apply box deltas (to
    normalize anchors that have un-normalized coordinate values). After this we convert the pre_nms_anchors back
    to the
    original shape of [num_images, num_anchors, (y1, x1,y2, x2)]

    :return:
    '''
    height = pre_nms_anchors[:, :, 2] - pre_nms_anchors[:, :, 0]
    width = pre_nms_anchors[:, :, 3] - pre_nms_anchors[:, :, 1]
    center_y = pre_nms_anchors[:, :, 0] + 0.5 * height
    center_x = pre_nms_anchors[:, :, 1] + 0.5 * width

    # Apply Box Delta (A)
    center_y += bbox_delta[:, :, 0] * height
    center_x += bbox_delta[:, :, 1] * width
    height *= tf.exp(bbox_delta[:, :, 2])
    width *= tf.exp(bbox_delta[:, :, 3])

    # Convert back to (y1, x1, y2, x2)
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width

    out = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    out = tf.transpose(out, [0, 2, 1])
    return out

def clip_boxes_to_01_per_batch(anchor_delta, window):
    """
    Clips Boxes within the range 0,1

    :param box_delta: The anchor per pixel position boxes for each batch with 4 pixel coordinates.
    :param window: THe min and max coordinates of window (We use this because our predictions should lie i 0,
    1 range)
    :return:

    The idea is pretty basic here:
        1. We split the coordinates.
        2. Check if they lie in the window range, if not make them lie
        3. Then concat them back to the original shape
    More over bring the box coordinate prediction to the range of [0,1] also helps us performing the next
    step i.e
    non-max suppression
    """

    # Window: [0,0,1,1] # 0,0 represents the top left corner and 1,1 represents the bottom right corner
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(anchor_delta, 4, axis=2)

    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)

    return tf.concat([y1, x1, y2, x2], axis=2, name="clipped_boxes")


class Proposals():
    '''
    The input to this network is:
    rpn_class_probs: [num_images, anchor, [back_ground_probability, fore_ground_probability]]
    '''
    
    def __init__(self, conf, batch_size, rpn_class_probs, rpn_bbox, input_anchors, run_batch=False, DEBUG=False):
      
        self.DEBUG = DEBUG
        
        if DEBUG:
            self.rpn_bbox_stddev = conf.RPN_BBOX_STDDEV
            self.num_box_before_nms = 5
            self.num_boxes_after_nms = 4
            self.iou_threshold = 0.3
            self.batch_size = batch_size
        else:
            
            self.rpn_bbox_stddev = conf.RPN_BBOX_STDDEV
            self.num_box_before_nms = conf.PRE_NMS_ROIS_INFERENCE  # 5
            self.num_boxes_after_nms = conf.POST_NMS_ROIS_INFERENCE  # 4
            self.iou_threshold = conf.RPN_NMS_THRESHOLD  # 0.3
            self.batch_size = batch_size

        rpn_class_probs = tf.cast(rpn_class_probs, dtype=tf.float32)
        rpn_bbox = tf.cast(rpn_bbox, dtype=tf.float32)
        input_anchors = tf.cast(input_anchors, dtype=tf.float32)
        
        print (rpn_class_probs.get_shape().as_list())
        print (rpn_bbox.get_shape().as_list())
        print (input_anchors.get_shape().as_list())
        
        
        if not run_batch:
            self.proposals = []
            for i in range(0, batch_size):
                self.proposals.append(self.build_per_image(rpn_class_probs[i], rpn_bbox[i], input_anchors[i]))
    
            self.proposals = tf.stack(self.proposals, axis=0)
        else:
            self.proposals = self.build_per_batch(rpn_class_probs, rpn_bbox, input_anchors)
        
    
    def build_per_image(self, rpn_class_probs, rpn_bbox, anchors):
        '''
        :param rpn_class_probs: [num_anchors,]  where each element is the prob score of the foreground
                                say for one image with 256x256 feature map and 3 anchors (dim rpn_class_probs)= (196608, 2)
        :param rpn_bbox:        [num_anchors, (y1, x1, y2, x2)]
                                say for one image with 256x256 feature map and 3 anchors (dim rpn_bbox)= (196608, 4)
        :param anchors:         [num_anchors, (y1, x1, y2, x2)]
        :return:
        '''
        # We would like to only capture the foreground class probabilities
        scores = tf.reshape(rpn_class_probs[:, 1], [-1,1])
        logging.info('Foreground_probs shape: %s', str(scores.shape))
        
        # Box deltas = [batch, num_rois, 4]
        bbox_delta = rpn_bbox * np.reshape(self.rpn_bbox_stddev, [1, 4])
        logging.info('bbox_delta shape: %s', str(bbox_delta.shape))
        
        # Get the anchors [None, 2]
        logging.info('anchors shape: %s', str(anchors.shape))
        
        # # Searching through lots of anchors can be time consuming. So we would select at most top 6000 of them for
        # # further processing [anchors = [num_images, num_anhcors, 4]]
        max_anc_before_nms = tf.minimum(self.num_box_before_nms, tf.shape(anchors)[0])
        # logging.info('max_anc_before_nms shape: %s', str(max_anc_before_nms))
        #
        # # Here we fetch the idx of the top 6000 anchors
        ix = tf.nn.top_k(tf.squeeze(scores), max_anc_before_nms, sorted=True, name="top_anchors").indices
        ix = tf.reshape(ix, [-1,1])
        # logging.info('ix shape: %s', str(ix.get_shape().as_list()))
        #
        # # Now that we have the idx of the top anchors we would want to only gather the data related pertaining to
        # # those idx. We would wanna gather foreground_prob and boxes only for the selected anchors.
        # # scores = tf.gather_nd(scores, ix)
        scores, bbox_delta, anchors = self.gather_data_for_idx(ix, scores, bbox_delta, anchors)
        
        # # return ixs, mesh, scores, boxes, anchors
        anchor_delta = apply_box_deltas_per_image(anchors, bbox_delta)
        #
        # # The boxes can have values at the interval of 0,1
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        anchor_delta_clipped = clip_boxes_to_01_per_image(anchor_delta=anchor_delta, window=window)
        #
        # # Perform Non-max suppression. Non max suppression is performed for one image at a time, so we loop over the
        # #  images here and stack them at the end
        proposals_per_image = self.non_max_suppression(
                                        scores = tf.squeeze(scores),
                                        proposals = anchor_delta_clipped,
                                        max_boxes = self.num_boxes_after_nms,
                                        iou_threshold = self.iou_threshold
                                )

       
        if self.DEBUG:
            self.ix = ix
            self.scores = scores
            self.bbox_delta = bbox_delta
            self.anchors = anchors
            self.anchor_delta = anchor_delta
            self.anchor_delta_clipped = anchor_delta_clipped
            
        return proposals_per_image
    
    
    def build_per_batch(self, rpn_class_probs, rpn_bbox, anchors):
        """
        Main function : required to get the filtered box (proposals)

        :param config:
        :param batch_size:
            inputs:
            (1, 196608, 2) (1, 196608, 2) (1, 196608, 4)
            * rpn_class_probs: [batch, anchors, (bg prob, fg prob)]
                        say for one image with 256x256 feature map and 3 anchors (dim rpn_class_probs)= (1,
                        196608, 2)
            * rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
                        say for one image with 256x256 feature map and 3 anchors (dim rpn_bbox)= (1, 196608, 4)
            * anchors: [batch, (y1, x1, y2, x2)] anchors in normalized coordinates
        :return:
        """
    
        # We would like to only capture the foreground class probabilities
        scores = rpn_class_probs[:, :, 1]
        logging.info('Foreground_probs shape: %s', str(scores.shape))
    
        # Box deltas = [batch, num_rois, 4]
        bbox_delta = rpn_bbox * np.reshape(self.rpn_bbox_stddev, [1, 1, 4])
        logging.info('bbox_delta shape: %s', str(bbox_delta.shape))
    
        # Get the anchors [None, 2]
        logging.info('anchors shape: %s', str(anchors.shape))
    
        # Searching through lots of anchors can be time consuming. So we would select at most top 6000 of them
        # for further processing [anchors = [num_images, num_anhcors, 4]]
        max_anc_before_nms = tf.minimum(self.num_box_before_nms, tf.shape(anchors)[1])
        logging.info('max_anc_before_nms shape: %s', str(max_anc_before_nms))
    
        # Here we fetch the idx of the top 6000 anchors
        ix = tf.nn.top_k(scores, max_anc_before_nms, sorted=True, name="top_anchors").indices
        logging.info('ix shape: %s', str(ix.get_shape().as_list()))
    
        # Now that we have the idx of the top anchors we would want to only gather the data related pertaining to
        # those idx. We would wanna gather foreground_prob and boxes only for the selected anchors.
        # scores = tf.gather_nd(scores, ix)
        mesh = tf.meshgrid(tf.range(tf.shape(ix)[1]), tf.range(tf.shape(ix)[0]))[1]
        ix = tf.stack([mesh, ix], axis=2)
        scores, bbox_delta, anchors = self.gather_data_for_idx(ix, scores, bbox_delta, anchors)
    
        # return ixs, mesh, scores, boxes, anchors
        anchor_delta = apply_box_deltas_per_batch(anchors, bbox_delta)
    
        # The boxes can have values at the interval of 0,1
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        anchor_delta_clipped = clip_boxes_to_01_per_batch(anchor_delta=anchor_delta, window=window)
    
        # Perform Non-max suppression. Non max suppression is performed for one image at a time, so we loop over the
        #  images here and stack them at the end
        proposals = tf.concat([
            tf.stack([
                self.non_max_suppression(scores[num],
                                         anchor_delta_clipped[num],
                                         max_boxes=self.num_boxes_after_nms,
                                         iou_threshold=self.iou_threshold)
            ], axis=0, name='nms_%s' % str(num)
            ) for num in range(0, self.batch_size)], axis=0, name='concat_boxes'
        )
    
        logging.info('bx_nw shape: %s', str(proposals.get_shape().as_list()))
    
        print('(Proposals) Proposals (shape) ', proposals.shape)

        if self.DEBUG:
            self.ix = ix
            self.scores = scores
            self.bbox_delta = bbox_delta
            self.anchors = anchors
            self.anchor_delta = anchor_delta
            self.anchor_delta_clipped = anchor_delta_clipped
            
        return proposals

    def non_max_suppression(self, scores, proposals, max_boxes=2, iou_threshold=0.7):
        """
        Applies Non-max suppression (NMS) to set of boxes

        Arguments:
        scores -- tensor of shape (None,),
        boxes -- tensor of shape (None, 4),  [y1, x1, y2, x2]  where (y1, x1) are diagonal coordinates to (y2, x2)
        max_boxes -- integer, maximum number of predicted boxes you'd like
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

        Returns:
        boxes -- tensor of shape (4, None), predicted box coordinates

        """
        
        # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
        nms_indices = tf.image.non_max_suppression(proposals,
                                                   scores,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_threshold,
                                                   name='activeBox_indice')
        
        proposals = tf.gather(proposals, nms_indices)
        
        # Sometimes due to the threshold set some batches may return num_proposals < num_boxes_after_nms
        # Such a case would make inconsitent proposal shape across different batches. Inorder to overcome this
        # problem, we pad the proposals with additional [0,0,0,0]
        padding = tf.maximum(self.num_boxes_after_nms - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals
    
    def gather_data_for_idx(self, ix, scores, bbox_delta, anchors):
        '''
        :param ix:           List [], indexes in order to be selected
        :param scores:       [num_anchors, 1]
        :param bbox_delta:
        :param anchors:
        :return:
        '''
        # Gather only the data pertaining to the ixs
        scores = tf.gather_nd(scores, ix)
        logging.info('scores shape = %s', str(scores.shape))
        
        # Gather only the data pertaining to the ixs
        bbox_delta = tf.gather_nd(bbox_delta, ix)
        logging.info('Box delta shape = %s', str(bbox_delta.shape))
        
        # Gather only the data pertaining to the ixs
        anchors = tf.gather_nd(anchors, ix)
        logging.info('anchors shape = %s', str(anchors.shape))
        
        return scores, bbox_delta, anchors
    
    def get_proposals(self):
        return self.proposals
    
    def debug_outputs(self):
        return (self.ix, self.scores, self.bbox_delta, self.anchors, self.anchor_delta, self.anchor_delta_clipped, self.proposals)


def debugg():
    from MaskRCNN_loop.config import config as conf
    
    np.random.seed(325)
    batch_size = 3
    a = np.array(np.random.random((batch_size, 5, 2)), dtype='float32')
    b = np.array(np.random.random((batch_size, 5, 4)), dtype='float32')
    c = np.array(np.random.random((batch_size, 5, 4)), dtype='float32')
    
    print('rpn_class_scores \n', a)
    print('rpn_bbox \n', b)
    print('anchors \n', c)
    
    
    obj_p = Proposals(conf, batch_size=3, rpn_class_probs=a, rpn_bbox=b, input_anchors=c, DEBUG=True)
    (ix, scores, bbox_delta, anchors, anchor_delta, anchor_delta_clipped, proposals) = \
        obj_p.debug_outputs()
    
    
   
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        (ix_, scores_, bbox_delta_, anchors_, anchor_delta_, anchor_delta_clipped_, proposals_)\
            = sess.run([ix, scores, bbox_delta, anchors, anchor_delta, anchor_delta_clipped,
                        proposals])
        print ('ix_ ', ix_)
        print('scores_ ', scores_.shape, scores_)
        print('')
        print('bbox_delta_ ', bbox_delta_.shape, bbox_delta_)
        print('')
        print('anchors_ ', anchors_.shape, anchors_)
        print('')
        print('anchor_delta_ ', anchor_delta_.shape, anchor_delta_)
        print('')
        print('anchor_delta_clipped_ ', anchor_delta_clipped_.shape, anchor_delta_clipped_)
        print('')
        print('proposals_ ', proposals_.shape, proposals_ )

# debugg()