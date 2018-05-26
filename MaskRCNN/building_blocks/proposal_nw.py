
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
import keras.engine as KE

from MaskRCNN.config import config as conf
logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


class ProposalLayer(KE.Layer):
    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        
    def call(self, inputs):
        foreground_probs = inputs[0][:, :, 1]
        logging.info('Foreground_probs shape: %s', str(foreground_probs.get_shape().as_list()))
    
        # Box deltas = [batch, num_rois, 4]
        box_delta = inputs[1] * np.reshape(conf.RPN_BBOX_STD_DEV, [1, 1, 4])
        logging.info('box_delta shape: %s', str(box_delta.get_shape().as_list()))
    
        # Get the anchors [None, 2]
        anchors = inputs[2]
        logging.info('anchors shape: %s', str(anchors.get_shape().as_list()))
    
        # Searching through lots of anchors can be time consuming. So we would select at most 6000 of them for further
        # processing
        max_anc_before_nms = tf.minimum(3, tf.shape(anchors)[1])
        logging.info('max_anc_before_nms shape: %s', str(max_anc_before_nms))
    
        # Here we fetch the idx of the top 6000 anchors
        ix = tf.nn.top_k(foreground_probs, max_anc_before_nms, sorted=True, name="top_anchors").indices
        logging.info('ix shape: %s', str(ix.get_shape().as_list()))
        # logging.info('ix[0] shape: %s', str(ix.get_shape().as_list()[1]))
        # logging.info('ix[1] shape: %s', str(ix.get_shape().as_list()[0]))
        # logging.info('%s ', tf.range(ix.get_shape().as_list()[1]).shape)
        # logging.info('%s ', tf.range(ix.get_shape().as_list()[0]).shape)
    
    
    
        # Now that we have the idx of the top anchors we would want to only gather the data related pertaining to those
        # idx. We would wanna gather foreground_prob and box_delta only for the selected anchors.

        # mesh = tf.meshgrid(tf.range(ix.get_shape().as_list()[1]),
        #                    tf.range(ix.get_shape().as_list()[0]))[1]

        mesh = tf.meshgrid(tf.range(tf.shape(ix)[1]),
                           tf.range(tf.shape(ix)[0]))[1]
        
        ixs = tf.stack([mesh, ix], axis=2)

        # Gather only the data pertaining to the ixs
        box_delta = tf.gather_nd(box_delta, ixs)
    
        # ixs = tf.constant([1,2])
    
        return mesh#[foreground_probs, box_delta, anchors, max_anc_before_nms, ix, ixs, mesh]
    
    def compute_output_shape(self, input_shape):
        logging.info('IN THE compute_output_shape function OF ProposalLayer')
        return (None, self.proposal_count, 4)
    
    
rpn_probs = tf.placeholder(dtype=tf.float32,
                               shape=[None, None, 2],
                               name="rpn_prob")

rpn_box = tf.placeholder(dtype=tf.float32,
                         shape=[None, None, 4],
                         name="rpn_box")

input_anchors = tf.placeholder(dtype=tf.float32,
                               shape=[None, None, 4],
                               name="input_anchors")
rpn_rois = ProposalLayer(
            proposal_count=1000,
            nms_threshold=0.7,
            name="ROI",
            config=conf)([rpn_probs, rpn_box, input_anchors])









def gather_data_for_idx(ix, box_delta):
    '''
    :param ix: Indexes of top 6000 anchors that have high foreground probability
    :param box_delta:
    :return:

    Say:

    Problem:
        ix = [[2,1,0],[0,1,3]] : this says the indices that are to be selected
        box_delta = [2, 5, 4]  2 (num batches), 5 (anchor_per_pixel_position), 4 (cx,cy,w,h)

    Solution:
        The idea is to select 3 anchor_per_pixel_position out of all the 5, so we need a output thats
        box_delta = [2,3,4], also we know that the 5 selection should be the indices 6,1,0
        This function is used to achieve it.

    How it works,
    Say box_delta = (2,5,4) [[[ 0.66850033  0.05690038  0.83834532  0.61043739]
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

    Then ixs = ixs_ = (2,3,2) [[[0 2]   # says to select the 2 index of 0 batch
                                [0 1]   # says to select the 2 index of 0 batch
                                [0 0]]  # says to select the 0 index of 0 batch

                              [[1 0]    # says to select the 0 index of 1 batch
                               [1 1]    # says to select the 1 index of 1 batch
                               [1 3]]]  # says to select the 3 index of 1 batch
    '''
    mesh = tf.meshgrid(tf.range(ix.get_shape().as_list()[1]),
                       tf.range(ix.get_shape().as_list()[0]))[1]
    ixs = tf.stack([mesh, ix], axis=2)
    
    
    # Gather only the data pertaining to the ixs
    box_delta = tf.gather_nd(box_delta, ixs)
    
    return ixs, mesh, box_delta


def proposals(conf, inputs):
    '''
    :param config:
    :param inputs:
        inputs:
        (1, 196608, 2) (1, 196608, 2) (1, 196608, 4)
        * rpn_probs: [batch, anchors, (bg prob, fg prob)]
                    say for one image with 256x256 feature map and 3 anchors (dim rpn_probs)= (1,196608, 2)
        * rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
                    say for one image with 256x256 feature map and 3 anchors (dim rpn_bbox)= (1, 196608, 4)
        * anchors: [batch, (y1, x1, y2, x2)] anchors in normalized coordinates
    :return:
    '''
    
    # We would like to only capture the foreground class probabilities
    foreground_probs = inputs[0][: ,: ,1]
    logging.info('Foreground_probs shape: %s', str(foreground_probs.get_shape().as_list()))
    
    # Box deltas = [batch, num_rois, 4]
    box_delta = inputs[1] * np.reshape(conf.RPN_BBOX_STD_DEV, [1, 1, 4])
    logging.info('box_delta shape: %s', str(box_delta.get_shape().as_list()))
    
    # Get the anchors [None, 2]
    anchors = inputs[2]
    logging.info('anchors shape: %s', str(anchors.get_shape().as_list()))
    
    # Searching through lots of anchors can be time consuming. So we would select at most 6000 of them for further
    # processing
    max_anc_before_nms = np.minimum(3, anchors.get_shape().as_list()[1])
    logging.info('max_anc_before_nms shape: %s', str(max_anc_before_nms))
    
    # Here we fetch the idx of the top 6000 anchors
    ix = tf.nn.top_k(foreground_probs, max_anc_before_nms, sorted=True, name="top_anchors").indices
    logging.info('ix shape: %s', str(ix.get_shape().as_list()))
    # logging.info('ix[0] shape: %s', str(ix.get_shape().as_list()[1]))
    # logging.info('ix[1] shape: %s', str(ix.get_shape().as_list()[0]))
    # logging.info('%s ', tf.range(ix.get_shape().as_list()[1]).shape)
    # logging.info('%s ', tf.range(ix.get_shape().as_list()[0]).shape)
    
    
    
    # Now that we have the idx of the top anchors we would want to only gather the data related pertaining to those
    # idx. We would wanna gather foreground_prob and box_delta only for the selected anchors.
    
    ixs, mesh, box_delta = gather_data_for_idx(ix, box_delta)
    # foreground_probs2 = tf.gather_nd(foreground_probs, ix)
    
    # ixs = tf.constant([1,2])
    
    return foreground_probs, box_delta, anchors, max_anc_before_nms, ix, ixs, mesh


def debugg():
    from MaskRCNN.config import config as conf
    
    rpn_probs = tf.placeholder(dtype=tf.float32,
                               shape=[2, 5, 2],
                               name="rpn_prob")
    
    rpn_box = tf.placeholder(dtype=tf.float32,
                             shape=[2, 5, 4],
                             name="rpn_box")
    
    
    input_anchors = tf.placeholder(dtype=tf.float32,
                                   shape=[1, 4, 4],
                                   name="input_anchors")
    
    # rpn_probs = tf.placeholder(dtype=tf.float32,
    #                            shape=[None, None, 2],
    #                            name="rpn_prob")
    #
    # rpn_box = tf.placeholder(dtype=tf.float32,
    #                          shape=[None, None, 4],
    #                          name="rpn_box")
    #
    # input_anchors = tf.placeholder(dtype=tf.float32,
    #                                shape=[None, None, 4],
    #                                name="input_anchors")
    
    foreground_probs, box_delta, anchors, max_anc_before_nms, ix, ixs, mesh = proposals(conf, inputs=[rpn_probs,
                                                                                                      rpn_box,
                                                                                                      input_anchors])
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a= np.random.random((2, 5, 2))
        b= np.random. random((2, 5, 4))
        c= np.random. random((1, 4, 4))
        feed_dict={rpn_probs: a, rpn_box: b, input_anchors: c}
        
        print('rpn_probs ', a)
        print('')
        print('rpn_box ', b)
        print('')
        print('input_anchors ', c)
        fp, bxd, anch, ix_, ixs_, mesh_ = sess.run([foreground_probs, box_delta, anchors, ix, ixs, mesh],
                                                   feed_dict=feed_dict)
        
        print ('')
        print('')
        print('foreground_probs = ', foreground_probs.shape, fp)
        print('')
        print('box_delta = ', bxd.shape, bxd)
        print('')
        print('anchors = ', anch.shape, anch)
        print('')
        print('ix_ = ', ix_)
        print('')
        print('mesh_ = ', mesh_)
        print('')
        print('ixs_ = ', ixs_)
        
        # print('')
        # print('fp2 = ', fp2)


# debugg()