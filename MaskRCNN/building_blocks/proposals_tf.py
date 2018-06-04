'''
Till this point We have already performed the FPN(feature pyramid network) and "Region Proposal Network". As an output from the RPN net we have:
    1. rpn_class_logits: [batch_size, pixel_position * num_anchor, 2]:
        This gives a binary outcome, if an anchor at a pixel for a image is foreground or background
    2. rpn_class_logits: [batch_size, pixel_position * num_anchor, 2]:
        This are just sigmoid outcomes of the Logits
    3. rpn_bbox: [batch_size, pixel_position * num_anchors, 4]:
        This outputs continuous values that outputs the bounding box of the anchors
        
Problem: For 1 pixel position we can have multiple anchors that can qualify as a bounding box for an object. Therefore in this module we take care of overlaps and select only the bounding box that has high IOU. This is also implemented using non-max supression.

'''


import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


class Proposals():
    '''
    The input to this network is:
    rpn_class_probs: [num_batches, anchor, [back_ground_probability, fore_ground_probability]]
    '''
    def __init__(self, conf, inference_batch_size):
        self.rpn_class_probs = tf.placeholder(dtype=tf.float32,
                                               shape=[None, None, 2],
                                               name="rpn_prob")
    
        self.rpn_bbox = tf.placeholder(dtype=tf.float32,
                                  shape=[None, None, 4],
                                  name="rpn_bbox")
    
        self.input_anchors = tf.placeholder(dtype=tf.float32,
                                       shape=[None, None, 4],
                                       name="input_anchors")
        self.rpn_bbox_stddev = conf.RPN_BBOX_STDDEV
        self.num_box_before_nms = 4 #conf.PRE_NMS_ROIS_INFERENCE
        self.num_boxes_after_nms = 2#conf.POST_NMS_ROIS_INFERENCE
        self.iou_threshold = 0.3#conf.RPN_NMS_THRESHOLD
        self.inference_batch_size = inference_batch_size
        
        self.build()

    def build(self):
        """
        Main function : required to get the filtered box (proposals)

        :param config:
        :param inference_batch_size:
            inputs:
            (1, 196608, 2) (1, 196608, 2) (1, 196608, 4)
            * rpn_class_probs: [batch, anchors, (bg prob, fg prob)]
                        say for one image with 256x256 feature map and 3 anchors (dim rpn_class_probs)= (1,196608, 2)
            * rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
                        say for one image with 256x256 feature map and 3 anchors (dim rpn_bbox)= (1, 196608, 4)
            * anchors: [batch, (y1, x1, y2, x2)] anchors in normalized coordinates
        :return:
        """
    
        # We would like to only capture the foreground class probabilities
        scores = self.rpn_class_probs[:, :, 1]
        logging.info('Foreground_probs shape: %s', str(scores.get_shape().as_list()))
    
        # Box deltas = [batch, num_rois, 4]
        bbox_delta = self.rpn_bbox * np.reshape(self.rpn_bbox_stddev, [1, 1, 4])
        logging.info('bbox_delta shape: %s', str(bbox_delta.get_shape().as_list()))
    
        # Get the anchors [None, 2]
        anchors = self.input_anchors
        logging.info('anchors shape: %s', str(anchors.get_shape().as_list()))
    
        # Searching through lots of anchors can be time consuming. So we would select at most top 6000 of them for
        # further
        # processing
        max_anc_before_nms = tf.minimum(self.num_box_before_nms, tf.shape(anchors)[1])
        logging.info('max_anc_before_nms shape: %s', str(max_anc_before_nms))
    
        # Here we fetch the idx of the top 6000 anchors
        ix = tf.nn.top_k(scores, max_anc_before_nms, sorted=True, name="top_anchors").indices
        logging.info('ix shape: %s', str(ix.get_shape().as_list()))
    
        # Now that we have the idx of the top anchors we would want to only gather the data related pertaining to
        # those idx. We would wanna gather foreground_prob and boxes only for the selected anchors.
        # scores = tf.gather_nd(scores, ix)
        scores = self.gather_data_for_idx(ix, scores, bbox_delta, anchors)
    
        # The boxes can have values at the interval of 0,1
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        self.clip_boxes_to_01(window=window)
    
        # Perform Non-max suppression. Non max suppression is performed for one image at a time, so we loop over the
        #  images here and stack them at the end
        self.proposals = tf.concat([
            tf.stack([
                self.non_max_suppression(scores[num],
                                         self.anchor_delta[num],
                                         max_boxes=self.num_boxes_after_nms,
                                         iou_threshold=self.iou_threshold)
            ], axis=0, name='nms_%s' % str(num)
            ) for num in range(0, self.inference_batch_size)], axis=0, name='concat_boxes'
        )
        logging.info('bx_nw shape: %s', str(self.proposals.get_shape().as_list()))
        
    def apply_box_deltas(self, pre_nms_anchors, bbox_delta):
        '''
        Applying Box Deltas to Anchors
        
        pre_nms_anchors = [num_batches, num_anchors, (y1, x1, y2, x2)]
        self.bbox_delta = [num_batches, num_anchors, (d(c_y), d(c_x), log(h), log(w))]
        
                    _____________ (x2, y2)
                    |           |
                    |           |
                    |           |
                    |           |
                    |           |
            (x1,y1) -------------
        
        Since our predictions are normalized and are in the form of [d(c_y), d(c_x), log(h), log(w)],
        we first convert our anchors to the form of [center_y, center_x, h, w] and then apply box deltas (to
        normalize anchors that have un-normalized coordinate values). After this we convert the pre_nms_anchors back to the
        original shape of [num_batches, num_anchors, (y1, x1,y2, x2)]
        
        :return:
        '''
        height = pre_nms_anchors[:,:,2] - pre_nms_anchors[:,:,0]
        width = pre_nms_anchors[:,:,3] - pre_nms_anchors[:,:,1]
        center_y = pre_nms_anchors[:,:,0] + 0.5*height
        center_x = pre_nms_anchors[:,:,1] + 0.5*width
        
        # Apply Box Delta (A)
        center_y += bbox_delta[:,:,0] * height
        center_x += bbox_delta[:,:,1] * width
        height *= tf.exp(bbox_delta[:,:,2])
        width *= tf.exp(bbox_delta[:,:,3])
        
        # Convert back to (y1, x1, y2, x2)
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        print (y1.shape, x1.shape, y2.shape, x2.shape)
        out = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
        out = tf.transpose(out, [0, 2, 1])
        return out

    def non_max_suppression(self, scores, boxes, max_boxes=2, iou_threshold=0.7):
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
        nms_indices = tf.image.non_max_suppression(boxes,
                                                   scores,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_threshold,
                                                   name='activeBox_indice')
    
        boxes = tf.gather(boxes, nms_indices)
        return boxes

    def clip_boxes_to_01(self, window):
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
        More over bring the box coordinate prediction to the range of [0,1] also helps us performing the next step i.e
        non-max suppression
        """
    
        # Window: [0,0,1,1] # 0,0 represents the top left corner and 1,1 represents the bottom right corner
        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(self.anchor_delta, 4, axis=2)
    
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    
        self.anchor_delta = tf.concat([y1, x1, y2, x2], axis=2, name="clipped_boxes")

    def gather_data_for_idx(self, ix, scores, bbox_delta, anchors):
        '''
        Gathers data given indexes

        :param ix: Indexes of top 6000 anchors that have high foreground probability
        :param boxes:
        :return:

        Say:

        Problem:
            ix = [[2,1,0],[0,1,3]] : this says the indices that are to be selected
            boxes = [2, 5, 4]  2 (num batches), 5 (anchor_per_pixel_position), 4 (cx,cy,w,h)

        Solution:
            The idea is to select 3 anchor_per_pixel_position out of all the 5, so we need a output thats
            boxes = [2,3,4], also we know that the 5 selection should be the indices 6,1,0
            This function is used to achieve it.

        How it works,
        Say boxes = (2,5,4) [[[ 0.66850033  0.05690038  0.83834532  0.61043739]
                                  [ 0.96072494  0.90195686  0.38814074  0.09934505]
                                  [ 0.70222181  0.64386777  0.27915297  0.76483525]
                                  [ 0.32436762  0.09989426  0.42256737  0.24381131]
                                  [ 0.35363515  0.45314872  0.19147657  0.49124077]]

                                 [[ 0.26162598  0.89599185  0.74032475  0.15512492]
                                  [ 0.44482893  0.65829518  0.99109874  0.38420606]
                                  [ 0.74626909  0.68953617  0.419537    0.73916023]
                                  [ 0.72346939  0.96696021  0.90526521  0.65514771]
                                  [ 0.10160118  0.89592455  0.11942481  0.7416876 ]]]
        Say ix = (2,3) = [[2 1 0]
                          [0 1 3]]

        Then tf.range(ix.get_shape().as_list()[1] = [0,1,2]
             tf.range(ix.get_shape().as_list()[1] = [0,1]

        Then mesh = (2, 3) [[0 0 0]
                            [1 1 1]]

        Then ixs =(2,3,2) [[[0 2]   # says to select the 2 index of 0 image
                            [0 1]   # says to select the 2 index of 0 image
                            [0 0]]  # says to select the 0 index of 0 image

                          [[1 0]    # says to select the 0 index of 1 image
                           [1 1]    # says to select the 1 index of 1 image
                           [1 3]]]  # says to select the 3 index of 1 image
        '''
        
        print ('Gathering Data')
        mesh = tf.meshgrid(tf.range(tf.shape(ix)[1]), tf.range(tf.shape(ix)[0]))[1]
        ixs = tf.stack([mesh, ix], axis=2)
    
        # Gather only the data pertaining to the ixs
        scores = tf.gather_nd(scores, ixs)
        logging.info('scores shape = %s', str(scores.shape))
    
        # Gather only the data pertaining to the ixs
        bbox_delta = tf.gather_nd(bbox_delta, ixs)
        logging.info('Box delta shape = %s', str(bbox_delta.shape))
    
        # Gather only the data pertaining to the ixs
        anchors = tf.gather_nd(anchors, ixs)
        logging.info('anchors shape = %s', str(anchors.shape))
    
        # return ixs, mesh, scores, boxes, anchors
        self.anchor_delta = self.apply_box_deltas(anchors, bbox_delta)
        
        return scores
    
    
    def get_proposal_graph(self):
        return dict(rpn_probs=self.rpn_class_probs, rpn_bbox=self.rpn_bbox,
                    input_anchors=self.input_anchors, proposals=self.proposals)
    
    def get_anchors_delta(self):
        return self.anchor_delta
   



def debugg():
    from MaskRCNN.config import config as conf

    np.random.seed(325)
    num_batches = 2
    proposal_count = 2
    nms_threshold = np.float32(0.3)

    a = np.array(np.random.random((3, 5, 2)), dtype='float32')
    b = np.array(np.random.random((3, 5, 4)), dtype='float32')
    c = np.array(np.random.random((3, 5, 4)), dtype='float32')

    obj_p = Proposals(conf, inference_batch_size=3)
    p_graph = obj_p.get_proposal_graph()
    # ix = obj_p.get_ix()
    # sc = obj_p.get_scores()
    # dt = obj_p.get_bboxes_delta()
    # anc = obj_p.get_anchors()
    ancd = obj_p.get_anchors_delta()
    # print(proposals)
    feed_dict = {p_graph['rpn_probs']: a, p_graph['rpn_bbox']: b, p_graph['input_anchors']: c}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # ix_ = sess.run([ix], feed_dict=feed_dict)
        # sc_ = sess.run([sc], feed_dict=feed_dict)
        # dt_ = sess.run([dt], feed_dict=feed_dict)
        # pnmsa_ = sess.run([anc], feed_dict=feed_dict)
        ancd_ = sess.run([ancd], feed_dict=feed_dict)
        p_ = sess.run(p_graph['proposals'], feed_dict=feed_dict)
        # print(ix_)
        # print ('')
        # print (sc_)
        # print('')
        # print(dt_[0].shape)
        # print('')
        # print(pnmsa_[0].shape)
        print('')
        print(ancd_)
        print ('')
        print(p_)


# debugg()




'''
ix = [array([[0, 3, 2, 1, 4],
       [1, 2, 0, 4, 3],
       [4, 0, 2, 3, 1]], dtype=int32)]

sc_ = [array([[ 0.98104852,  0.79038447,  0.76019788,  0.68306577],
       [ 0.95836937,  0.92400974,  0.55117744,  0.54641497],
       [ 0.92985797,  0.9152084 ,  0.90162092,  0.88823181]], dtype=float32)]
       
dt_ = [array([[[  6.08330965e-02,   6.71194121e-02,   9.41824615e-02,
           1.71722159e-01],
        [  3.85359898e-02,   9.32771638e-02,   6.52525723e-02,
           1.92653507e-01],
        [  6.77995607e-02,   6.24737255e-02,   1.89149845e-02,
           1.74794272e-02],
        [  8.25559869e-02,   4.15760539e-02,   1.90824702e-01,
           7.37159774e-02]],

       [[  1.43593876e-02,   1.00482712e-02,   5.03815077e-02,
           3.92356282e-03],
        [  7.38601610e-02,   2.49234634e-03,   1.27579302e-01,
           6.00859821e-02],
        [  5.37443422e-02,   3.63819599e-02,   1.18435375e-01,
           7.35748857e-02],
        [  5.09575866e-02,   8.80408734e-02,   1.45163164e-01,
           1.63295969e-01]],

       [[  9.55238715e-02,   1.13850981e-02,   1.28908351e-01,
           2.66324226e-02],
        [  7.86226615e-02,   9.94997323e-02,   1.70552894e-01,
           7.04915524e-02],
        [  9.70137939e-02,   7.02699199e-02,   1.12953506e-01,
           4.68629441e-05],
        [  1.20924581e-02,   2.32591256e-02,   1.77218363e-01,
           1.01405002e-01]]], dtype=float32)]
           
pnmsa_ = [array([[[ 0.66516078,  0.7107172 ,  0.104709  ,  0.41347158],
        [ 0.4026624 ,  0.00647369,  0.97270262,  0.70907563],
        [ 0.36219054,  0.18682894,  0.75377899,  0.75660789],
        [ 0.0971365 ,  0.30265555,  0.30198509,  0.8906796 ]],

       [[ 0.51335049,  0.67271036,  0.55130559,  0.13511348],
        [ 0.45991206,  0.69495296,  0.141526  ,  0.19375683],
        [ 0.71754569,  0.79186046,  0.89333463,  0.11504316],
        [ 0.17198341,  0.0888403 ,  0.83701622,  0.88303244]],

       [[ 0.40544087,  0.96145767,  0.37492931,  0.86902213],
        [ 0.63724017,  0.69959635,  0.14438523,  0.45761779],
        [ 0.50528699,  0.13827209,  0.50424409,  0.81720257],
        [ 0.2975572 ,  0.38713291,  0.40366048,  0.97275996]]], dtype=float32)]
        
anchor_delta = [array([[[ 0.65874195,  0.71861047,  0.04293984,  0.36567649],
        [ 0.40541095, -0.00262791,  1.01388812,  0.84925073],
        [ 0.38500136,  0.21740168,  0.78406721,  0.79722756],
        [ 0.09251355,  0.30461103,  0.34043097,  0.93761951]],

       [[ 0.5129149 ,  0.66836518,  0.55283123,  0.12865484],
        [ 0.45805818,  0.70922279,  0.09634772,  0.17698866],
        [ 0.71594203,  0.79307377,  0.91383362,  0.06458205],
        [ 0.15392342,  0.08832273,  0.92285311,  1.02339268]],

       [[ 0.40462527,  0.96165276,  0.36991575,  0.86672235],
        [ 0.64431632,  0.68435603,  0.05980986,  0.42470446],
        [ 0.50524819,  0.18596455,  0.50408059,  0.86492682],
        [ 0.28855395,  0.36950359,  0.41522977,  1.01763153]]], dtype=float32)]
        
[array([[[ 0.65874195,  0.71861047,  0.04293984,  0.36567649],
        [ 0.40541095,  0.        ,  1.        ,  0.84925073],
        [ 0.38500136,  0.21740168,  0.78406721,  0.79722756],
        [ 0.09251355,  0.30461103,  0.34043097,  0.93761951]],

       [[ 0.5129149 ,  0.66836518,  0.55283123,  0.12865484],
        [ 0.45805818,  0.70922279,  0.09634772,  0.17698866],
        [ 0.71594203,  0.79307377,  0.91383362,  0.06458205],
        [ 0.15392342,  0.08832273,  0.92285311,  1.        ]],

       [[ 0.40462527,  0.96165276,  0.36991575,  0.86672235],
        [ 0.64431632,  0.68435603,  0.05980986,  0.42470446],
        [ 0.50524819,  0.18596455,  0.50408059,  0.86492682],
        [ 0.28855395,  0.36950359,  0.41522977,  1.        ]]], dtype=float32)]
'''

