

import numpy as np
import tensorflow as tf
from MaskRCNN.building_blocks.utils import norm_boxes
from MaskRCNN.building_blocks.proposals_tf import apply_box_deltas, clip_boxes_to_01


class DetectionLayer():
    def __init__(self, conf, image_shape, window, proposals, mrcnn_class_probs, mrcnn_bbox, DEBUG=False):
        self.bbox_stddev = conf.BBOX_STD_DEV
        self.detection_post_nms_instances = conf.DETECTION_POST_NMS_INSTANCES
        self.detection_min_thresh = conf.DETECTION_MIN_THRESHOLD
        self.detection_nms_threshold = conf.DETECTION_NMS_THRESHOLD
        
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

        # LOOP FOR EACH BATCH : An Expensive Operation
        for i in range(0,1):
            # We have total of 81 classes where the class_id (0) is the background prediction, so we select boxes where
            # class_id is greater that 0
            object_ids = tf.where(class_ids[i] > 0)[:,0] # Outputs a list

            # Select only class detection that have more than 0.7 class_scores in finding an object
            object_ids_more_07 = tf.where(class_scores[i] > self.detection_min_thresh)[:,0]  # Outputs a list

            cmn_object_ids = tf.sets.set_intersection(tf.expand_dims(object_ids, 0), tf.expand_dims(
                    object_ids_more_07, 0))
            cmn_object_ids = tf.reshape(tf.sparse_tensor_to_dense(cmn_object_ids), [-1,1])

            pre_nms_class_ids = tf.gather_nd(tf.squeeze(class_ids[i]), cmn_object_ids)
            pre_nms_scores = tf.gather_nd(tf.squeeze(class_scores[i]), cmn_object_ids)
            pre_nms_proposals = tf.gather_nd(tf.squeeze(refined_proposals[i]), cmn_object_ids)
            unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

            # GET THE PROPOSAL INDEX FOR EACH CLASS
            def get_post_nms_proposal_ids_for_a_class(class_id):
                idx = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
                idx = tf.reshape(idx, [-1, 1])
    
                idx_nms_ids = tf.image.non_max_suppression(
                        boxes=tf.gather_nd(pre_nms_proposals, idx),
                        scores=tf.gather_nd(pre_nms_scores, idx),
                        max_output_size=self.detection_post_nms_instances,
                        iou_threshold=self.detection_nms_threshold
                )
    
                # class_keep_idx = tf.gather(cmn_object_ids, )
                class_nms_idx = tf.squeeze(tf.gather(cmn_object_ids,
                                                     tf.gather_nd(idx, tf.reshape(idx_nms_ids, [-1, 1]))),
                                           [-1])
    
                # Sometimes some batches may have less number of proposals than the maximum number of post_nms detection
                # proposls. This discrepancy may result in different num_proposals (detection) for different batches
                # which  would give an error while concatenating batches. So we pad with zeros to make "each batch"
                # result in [detection_post_nms_instances, 4]
                extra_proposals_to_add = self.detection_post_nms_instances - tf.shape(class_nms_idx)[0]
                class_nms_idx = tf.squeeze(tf.pad(class_nms_idx, [[0, extra_proposals_to_add], [0, 0]], mode='CONSTANT',
                                                  constant_values=-1), [-1])
                # Suppose we find only 2 proposals at index (1,3) after nms and max is 5 then the output will be [1,
                # 3,-1,-1,-1]
                class_nms_idx.set_shape([self.detection_post_nms_instances])
    
                return class_nms_idx
            
            # Get post_nms_class_ids for a batch
            post_nms_class_id_idx = tf.map_fn(get_post_nms_proposal_ids_for_a_class, unique_pre_nms_class_ids, dtype=tf.int64)
            
            # Stack data for each class id in one list and remove -1 padding
            post_nms_class_id_idx = tf.reshape(post_nms_class_id_idx, [-1])  # Flatten
            post_nms_class_id_idx = tf.gather_nd(post_nms_class_id_idx, tf.where(post_nms_class_id_idx > -1))
            
            # FOR a batch we should only have only "detection_post_nms_instances" detections. We select only the top
            # detection using the scores
            

        
        
        if self.DEBUG:
            self.class_ids = class_ids
            self.indices = indices
            self.mesh = mesh
            self.ixs = ixs
            self.class_scores = class_scores
            self.bbox_delta = bbox_delta
            self.refined_proposals = refined_proposals
            self.object_ids = object_ids
            self.object_ids_more_07 = object_ids_more_07
            self.cmn_object_ids = cmn_object_ids
            self.pre_nms_class_ids = pre_nms_class_ids
            self.pre_nms_scores = pre_nms_scores
            self.pre_nms_porposals = pre_nms_proposals
            self.unique_pre_nms_class_ids = unique_pre_nms_class_ids
            self.class_nms_idx = []
            self.post_nms_class_id_idx = post_nms_class_id_idx

    
    
    
    def gather_idx(self):
        pass



    def debug_outputs(self):
        return (self.class_ids,
            self.indices,
            self.mesh,
            self.ixs,
            self.class_scores,
            self.bbox_delta,
            self.refined_proposals,
            self.object_ids,
            self.object_ids_more_07,
            self.cmn_object_ids,
            self.pre_nms_class_ids,
            self.pre_nms_scores,
            self.pre_nms_porposals,
            self.unique_pre_nms_class_ids,
            self.class_nms_idx,
            self.post_nms_class_id_idx
    )
    
    



def debug():
    from MaskRCNN.config import config as conf
    np.random.seed(863)
    proposals = np.array(np.random.random((2,8,4)), dtype='float32')
    mrcnn_class_probs = np.array(np.random.random((2,8,4)), dtype='float32') #[num_batch, num_top_proposal, num_classes]
    mrcnn_bbox = np.array(np.random.random((2,8,4,4)), dtype='float32')
    window = np.array([131, 0, 893, 1155], dtype='int32')  # image without zeropad [y1, x1, y2, x2]
    
    print('mrcnn_class_probs ', mrcnn_class_probs.shape, mrcnn_class_probs)
    print ('')
    print('mrcnn_bbox ', mrcnn_bbox)
    print('')

    (class_ids, indices, mesh, ixs, class_scores, bbox_delta, refined_proposals, object_ids, object_ids_more_07,
     cmn_object_ids, pre_nms_class_ids, pre_nms_scores, pre_nms_porposals, unique_pre_nms_class_ids, class_nms_idx, post_nms_class_id_idx
     )= DetectionLayer(
            conf, [1024, 1024, 3], window, proposals, mrcnn_class_probs, mrcnn_bbox, DEBUG=True).debug_outputs()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        (class_ids_, indices_, mesh_, ixs_, class_scores_, bbox_delta_, refined_proposals_, object_ids_,
         object_ids_more_07_, cmn_object_ids_, pre_nms_class_ids_, pre_nms_scores_, pre_nms_porposals_,
         unique_pre_nms_class_ids_, class_nms_idx_, post_nms_class_id_idx_)=  sess.run(
                [class_ids, indices, mesh, ixs, class_scores, bbox_delta, refined_proposals, object_ids, object_ids_more_07,
     cmn_object_ids, pre_nms_class_ids, pre_nms_scores, pre_nms_porposals, unique_pre_nms_class_ids, class_nms_idx, post_nms_class_id_idx]
            )#,
        # class_scores,
        #  bbox])
        print('class_ids_ ', class_ids_.shape, class_ids_)
        print ('')
        print('indices_ ', indices_)
        print('')
        print ('mesh_ ', mesh_)
        print('')
        print('ixs_', ixs_.shape, ixs_)
        print('')
        print('class_scores_ ', class_scores_.shape, class_scores_)
        print('')
        print('bbox_delta_ ',bbox_delta_.shape, bbox_delta_)
        print('')
        print('refined_proposals_ ', refined_proposals_.shape, refined_proposals_)
        print('')
        print('object_ids_ ', object_ids_)
        print('')
        print('object_ids_more_07_ ', object_ids_more_07_)
        print('')
        print('cmn_object_ids_ ', cmn_object_ids_)
        print('')
        print('pre_nms_class_ids_ ', pre_nms_class_ids_.shape, pre_nms_class_ids_)
        print('')
        print('pre_nms_scores_ ', pre_nms_scores_.shape, pre_nms_scores_)
        print('')
        print('pre_nms_porposals_ ', pre_nms_porposals_.shape, pre_nms_porposals_)
        print('')
        print('unique_pre_nms_class_ids_ ', unique_pre_nms_class_ids_.shape, unique_pre_nms_class_ids_)
        print('')
        print('class_nms_idx_ ', class_nms_idx_)
        print('')
        print('post_nms_class_id_idx_ ', post_nms_class_id_idx_.shape, post_nms_class_id_idx_)



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
