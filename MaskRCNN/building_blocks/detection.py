

import numpy as np
import tensorflow as tf
from MaskRCNN.building_blocks.utils import norm_boxes
from MaskRCNN.building_blocks.proposals_tf import apply_box_deltas, clip_boxes_to_01


class DetectionLayer():
    def __init__(self, conf, image_shape, window, proposals, mrcnn_class_probs, mrcnn_bbox, DEBUG=False):
        self.bbox_stddev = conf.BBOX_STD_DEV
        self.detection_min_thresh = conf.DETECTION_MIN_THRESHOLD
        
        self.DEBUG = DEBUG
        window = norm_boxes(window, image_shape[:2])  # Normalize the Image Window coordinates
        print ('window ', window)
        
        # Re modify data types
        window = tf.cast(window, dtype=tf.float32)
        proposals = tf.cast(proposals, dtype=tf.float32)
        mrcnn_class_probs = tf.cast(mrcnn_class_probs, dtype=tf.float32)
        mrcnn_bbox = tf.cast(mrcnn_bbox, dtype=tf.float32)
        
        self.build(window, proposals, mrcnn_class_probs, mrcnn_bbox)
        
    
    def build(self, window, proposals, mrcnn_class_probs, mrcnn_bbox):
        ''' What's going on:
        
        Section 1: This section is aimed to build a matrix of indices such that we can perform Slice and dice
        operation to gather mrcnn_class_probs and mrcnn_bbox for all the batches in one shot without looping through
        them:
        Example:
                if mrcnn_class_probs  (2, 3, 4): [[[ 0.77371078  0.39908367  0.59822175  0.36256842]
                                                   [ 0.33876734  0.56173228  0.74943795  0.53174218]
                                                   [ 0.94958731  0.42413447  0.51827236  0.30986127]]
                                                
                                                  [[ 0.77363264  0.16883411  0.38540116  0.26045984]
                                                   [ 0.72828903  0.5177639   0.33214877  0.33706013]
                                                   [ 0.88052308  0.83476073  0.4034089   0.66374143]]]
                
                then class_ids           (2, 3): [[0 2 0]
                                                  [0 0 0]]
                
                and  mesh                (2, 3): [[0 0 0]
                                                  [1 1 1]]
                                                  
                and indices              (2, 3): [[0 1 2]
                                                  [0 1 2]]
                                                  
                and ixs               (2, 3, 4): [[[0 0 0]    # Says to select batch(0), anchor(0) and class(0)
                                                   [0 1 2]    # Says to select batch(0), anchor(1) and class(2)
                                                   [0 2 0]]   # Says to select batch(0), anchor(2) and class(0)
                                                
                                                  [[1 0 0]    # Says to select batch(1), anchor(0) and class(0)
                                                   [1 1 0]    # Says to select batch(1), anchor(1) and class(0)
                                                   [1 2 0]]]  # Says to select batch(1), anchor(2) and class(0)
                                                   
        Section 2: This section just gathers the data that is to be processed
                                                  
        '''
        class_ids = tf.argmax(mrcnn_class_probs, axis=2, output_type=tf.int32)

        bbox_delta = mrcnn_bbox * self.bbox_stddev
        
        # Section 1:
        mesh = tf.meshgrid(tf.range(tf.shape(class_ids)[1]), tf.range(tf.shape(class_ids)[0]))[1]
        indices = tf.reshape(
                        tf.tile(tf.range(tf.shape(mrcnn_class_probs)[1]),
                                [tf.shape(class_ids)[0]]),
                tf.shape(class_ids)
        )
        ixs = tf.stack([mesh, indices, class_ids], axis=2)

        # Section 2:
        class_scores = tf.gather_nd(mrcnn_class_probs, ixs)
        bbox_delta  = tf.gather_nd(bbox_delta, ixs)

        # Apply Box Delta
        refined_proposals = apply_box_deltas(proposals, bbox_delta)
        
        # Clip image to image window, this time we dont do it on the total normed image coordinate, because our
        # original image may actually be zero padded.
        refined_proposals = clip_boxes_to_01(refined_proposals, window)

        # object_ids = tf.where(class_ids > 0)
        # object_ids_more_07 = tf.where(class_scores> self.detection_min_thresh)

        for i in range(0,1):
            # We have total of 81 classes where the class_id (0) is the background prediction, so we select boxes where
            # class_id is greater that 0
            object_ids = tf.where(class_ids[i] > 0)[:,0] # Outputs a list

            # Select only class detection that have more than 0.7 class_scores in finding an object
            object_ids_more_07 = tf.where(class_scores[i] > self.detection_min_thresh)[:,0]  # Outputs a list

            cmn_objects = tf.sets.set_intersection(tf.expand_dims(object_ids, 0), tf.expand_dims(
                    object_ids_more_07, 0))
            cmn_objects = tf.reshape(tf.sparse_tensor_to_dense(cmn_objects), [-1,1])

            pre_nms_class_ids = tf.gather_nd(tf.squeeze(class_ids[i]), cmn_objects)
            pre_nms_scores = tf.gather_nd(tf.squeeze(class_scores[i]), cmn_objects)
            pre_nms_porposals = tf.gather_nd(tf.squeeze(refined_proposals[i]), cmn_objects)
            unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        # Get the intersection of object_ids and object_ids_more_07
        # cmn_objects = tf.expand_dims(object_ids_more_07, 0)#tf.expand_dims(object_ids, 0)#tf.sets.set_intersection(
        # tf.expand_dims(object_ids, 0),
                                                # tf.expand_dims(object_ids_more_07, 0))

        
        
        if self.DEBUG:
            self.class_ids = class_ids
            self.indices = indices
            self.mesh = mesh
            self.ixs = ixs
            self.class_scores = class_scores
            self.bbox_delta = bbox_delta
            self.refined_proposals = refined_proposals
            self.clipped_proposals = []
            self.object_ids = object_ids
            self.object_ids_more_07 = object_ids_more_07
            self.cmn_objects = cmn_objects
            self.clipped_proposals2 = unique_pre_nms_class_ids

    
    def non_max_suppression(self):
        pass
    
    def gather_idx(self):
        pass



    def debug_outputs(self):
        return self.class_ids, self.indices, self.mesh, self.ixs, self.class_scores, self.bbox_delta, \
               self.refined_proposals, self.clipped_proposals, self.object_ids, self.object_ids_more_07, \
               self.cmn_objects, self.clipped_proposals2
    
    



def debug():
    from MaskRCNN.config import config as conf
    np.random.seed(863)
    proposals = np.array(np.random.random((2,3,4)), dtype='float32')
    mrcnn_class_probs = np.array(np.random.random((2,3,4)), dtype='float32') #[num_batch, num_top_proposal, num_classes]
    mrcnn_bbox = np.array(np.random.random((2,3,4,4)), dtype='float32')
    window = np.array([131, 0, 893, 1155], dtype='int32')  # image without zeropad [y1, x1, y2, x2]
    
    print('mrcnn_class_probs ', mrcnn_class_probs.shape, mrcnn_class_probs)
    print ('')
    print('mrcnn_bbox ', mrcnn_bbox)
    print('')
    
    
    class_ids, indices, mesh, ixs, class_scores, bbox_delta, refined_proposals, clipped_proposals, object_ids, \
    object_ids_more_07, cmn_objects, clipped_proposals2 \
        = DetectionLayer(
            conf, [1024, 1024, 3], window, proposals, mrcnn_class_probs, mrcnn_bbox, DEBUG=True).debug_outputs()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        class_ids_, indices_, mesh_, ixs_, class_scores_, bbox_delta_, refined_proposals_, clipped_proposals_, \
        object_ids_, object_ids_more_07_, cmn_objects_, clipped_proposals2_ = \
            sess.run(
                [class_ids, indices, mesh, ixs, class_scores, bbox_delta, refined_proposals, clipped_proposals,
                 object_ids, object_ids_more_07, cmn_objects, clipped_proposals2]
            )#,
        # class_scores,
        #  bbox])
        print('class_ids_ ', class_ids_.shape, class_ids_)
        print ('')
        print('indices_ ', indices_)
        print('')
        print ('mesh_ ', mesh_)
        print('')
        print(ixs_)
        print('')
        print('class_scores_ ', class_scores_.shape, class_scores_)
        print('')
        print('bbox_delta_ ',bbox_delta_.shape, bbox_delta_)
        print('')
        print('refined_proposals_ ', refined_proposals_.shape, refined_proposals_)
        print('')
        print('clipped_proposals_ ', clipped_proposals_)
        print('')
        print('object_ids_ ', object_ids_)
        print('')
        print('object_ids_more_07_ ', object_ids_more_07_)
        print('')
        print('cmn_objects_ ', cmn_objects_)
        print('')
        print('clipped_proposals2_ ', clipped_proposals2_.shape, clipped_proposals2_)

debug()
    
#
#
# bbox_delta_  (2, 3, 4) [[[ 0.08170921  0.08602113  0.14487155  0.11414423]
#   [ 0.05364669  0.06273616  0.14014252  0.04939182]
#   [ 0.07942625  0.01017553  0.11778816  0.01741946]]
#
#  [[ 0.06138347  0.05228876  0.16826003  0.12148649]
#   [ 0.00441313  0.08993775  0.04278338  0.08515557]
#   [ 0.00188644  0.05279312  0.16337026  0.18715716]]]
