
import numpy as np
import tensorflow as tf
from MaskRCNN.building_blocks.utils import norm_boxes, denorm_boxes
from MaskRCNN.building_blocks.proposals_tf import apply_box_deltas, clip_boxes_to_01


def unmold_detection(original_image_shape, image_shape, detections, image_window):
    '''
    :param detections: [batch_size, num_proposals, (y1, x1, y2, x2, class_id, class_score)]
    :return:`
    '''
    print('image_shape ', image_shape)
    print('original_image_shape ', original_image_shape)
    print('image_window ', image_window)

    image_window = norm_boxes(image_window, image_shape[:2])
    # print ('image_window_normeed: ', image_window)
    
    # print (detections.shape)
    zero_ix = np.where(detections[:, 4] == 0)[0]
    # print (zero_ix)
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
    # print (N)
    
    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]

    # Convert the Normalized coordinates to Pixel coordinates
    wy1, wx1, wy2, wx2 = image_window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # print ('Shift scale: ', shift, scale)
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # print ('boxes ', boxes)
    # Convert boxes to pixel coordinates on the original image
    boxes = denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)

    return boxes, class_ids, scores


class DetectionLayer():
    def __init__(self, conf, image_shape, num_batches, window, proposals, mrcnn_class_probs, mrcnn_bbox, DEBUG=False):
        self.image_shape = image_shape
        self.num_batches = num_batches
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
        
        self.detections = self.build(window, proposals, mrcnn_class_probs, mrcnn_bbox)
        # from MaskRCNN.building_blocks.detection2 import DetectionLayer2
        # self.detections = DetectionLayer2(conf, [1024, 1024, 3], window, proposals, mrcnn_class_probs,
        #                              mrcnn_bbox).get_detections()

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
        
        # LOOP FOR EACH BATCH : An Expensive Operation
        detections = []
        if self.DEBUG:
            pre_nms_proposals_list = []
            pre_nms_class_ids_list = []
            pre_nms_scores_list = []
            clipped_proposals_list = []
        
        for i in range(0,self.num_batches):
            # Clip image to image window, this time we dont do it on the total normed image coordinate, because our
            # original image may actually be zero padded.
            # refined_proposals = clip_boxes_to_01(refined_proposals, window)
            clipped_proposals = clip_boxes_to_01(tf.expand_dims(refined_proposals[i], 0), window[i])
            # print (clipped_proposals.get_shape().as_list())
            
            # We have total of 81 classes where the class_id (0) is the background prediction, so we select boxes where
            # class_id is greater that 0
            class_id_idx = tf.where(class_ids[i] > 0)[: ,0] # Outputs a list
            
            # Select only class detection that have more than 0.7 class_scores in finding an object
            score_id_idx = tf.where(class_scores[i] > self.detection_min_thresh)[: ,0]  # Outputs a list
            
            keep_idx = tf.sets.set_intersection(tf.expand_dims(class_id_idx, 0), tf.expand_dims( score_id_idx ,0))
            keep_idx = tf.reshape(tf.sparse_tensor_to_dense(keep_idx), [-1 ,1])
            
            pre_nms_class_ids = tf.gather_nd(tf.squeeze(class_ids[i]), keep_idx)
            pre_nms_scores = tf.gather_nd(tf.squeeze(class_scores[i]), keep_idx)
            pre_nms_proposals = tf.gather_nd(tf.squeeze(clipped_proposals), keep_idx)
            
            unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

            if self.DEBUG:
                
                clipped_proposals_list.append(clipped_proposals)
                pre_nms_proposals_list.append(pre_nms_proposals)
                pre_nms_class_ids_list.append(pre_nms_class_ids)
                pre_nms_scores_list.append(pre_nms_scores)
            
            # GET THE PROPOSAL INDEX FOR EACH CLASS
            def get_post_nms_proposal_ids_for_a_class(class_id):
                pre_nms_class_id_idx = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
                pre_nms_class_id_idx = tf.reshape(pre_nms_class_id_idx, [-1, 1])
                
                nms_idx = tf.image.non_max_suppression(
                        boxes=tf.gather_nd(pre_nms_proposals, pre_nms_class_id_idx),
                        scores=tf.gather_nd(pre_nms_scores, pre_nms_class_id_idx),
                        max_output_size=self.detection_post_nms_instances,
                        iou_threshold=self.detection_nms_threshold
                )
                
                # class_keep_idx = tf.gather(keep_idx, )
                post_nms_keep_idx = tf.squeeze(
                        tf.gather(keep_idx, tf.gather_nd(pre_nms_class_id_idx, tf.reshape(nms_idx, [-1, 1]))), [-1]
                )
                
                # Sometimes some batches may have less number of proposals than the maximum number of post_nms detection
                # proposls. This discrepancy may result in different num_proposals (detection) for different batches
                # which  would give an error while concatenating batches. So we pad with zeros to make "each batch"
                # result in [detection_post_nms_instances, 4]
                extra_proposals_to_add = self.detection_post_nms_instances - tf.shape(post_nms_keep_idx)[0]
                post_nms_keep_idx = tf.squeeze \
                    (tf.pad(post_nms_keep_idx, [[0, extra_proposals_to_add], [0, 0]], mode='CONSTANT',
                                                      constant_values=-1), [-1])
                # Suppose we find only 2 proposals at index (1,3) after nms and max is 5 then the output will be [1,
                # 3,-1,-1,-1]
                post_nms_keep_idx.set_shape([self.detection_post_nms_instances])
                
                return post_nms_keep_idx
            
            # Get post_nms_class_ids for a batch
            post_nms_keep_idx = tf.map_fn(get_post_nms_proposal_ids_for_a_class, unique_pre_nms_class_ids, dtype=tf.int64)
            
            # Stack data for each class id in one list and remove -1 padding
            post_nms_keep_idx = tf.reshape(post_nms_keep_idx, [-1])  # Flatten, (1,100) to (100,)
            post_nms_keep_idx = tf.gather_nd(post_nms_keep_idx, tf.where(post_nms_keep_idx > -1))
            
            # Compute the intersection between keep_idx and post_nms_keep_idx. Since pose_nms_keep_idx is derived
            # after fitering though keep_idx, this step may seem irrelevant.
            # keep_idx = [N,1] and post_nms_keep_idx = [N,], so we first reshape
            post_nms_keep_idx = tf.sets.set_intersection(tf.expand_dims(tf.reshape(keep_idx, [-1]), 0),
                                                         tf.expand_dims(post_nms_keep_idx, 0))
            post_nms_keep_idx = tf.sparse_tensor_to_dense(post_nms_keep_idx)[0]
            
            # FOR a batch we should only have max "100" detections. We select only the top detection using the scores
            post_nms_scores = tf.gather(tf.squeeze(class_scores[i]), post_nms_keep_idx)
            num_keep = tf.minimum(self.detection_post_nms_instances, tf.shape(post_nms_scores)[0])
            post_nms_topk_keep_idx = tf.gather(post_nms_keep_idx,
                                               tf.nn.top_k(post_nms_scores, k=num_keep, sorted=True)[1])
            
            
            # Arrange the output in the form of [num_proposals, (y1, x1, y2, x2), class_id, scores]
            # detection = tf.gather(tf.squeeze(class_scores[i]), post_nms_topk_keep_idx)
            detection_per_batch = tf.concat(
                    [tf.gather(tf.squeeze(clipped_proposals), post_nms_topk_keep_idx),
                     tf.reshape(tf.to_float(tf.gather(tf.squeeze(class_ids[i]), post_nms_topk_keep_idx)), [-1 ,1]),
                     tf.reshape(tf.gather(tf.squeeze(class_scores[i]), post_nms_topk_keep_idx), [-1 ,1])],
                    axis=1)
            
            # We may still end up with some batches having less than 100 instances, In those cases we would zero pad
            # the instances to make it to 100 count.
            num_instances_to_add = self.detection_post_nms_instances - tf.shape(detection_per_batch)[0]
            detections.append(tf.pad(detection_per_batch, [(0, num_instances_to_add), (0, 0)], "CONSTANT"))

        detections = tf.stack(detections, axis=0)
        
        if self.DEBUG:
            self.class_ids = class_ids
            self.indices = indices
            self.mesh = mesh
            self.ixs = ixs
            self.class_scores = class_scores
            self.bbox_delta = bbox_delta
            self.refined_proposals = refined_proposals
            # self.class_id_idx = class_id_idx
            # self.score_id_idx = score_id_idx
            # self.keep_idx = keep_idx
            self.clipped_proposals_list = clipped_proposals_list
            self.pre_nms_class_ids_list = pre_nms_class_ids_list
            self.pre_nms_scores_list = pre_nms_scores_list
            self.pre_nms_proposals_list = pre_nms_proposals_list
            # self.unique_pre_nms_class_ids = unique_pre_nms_class_ids
            # self.post_nms_keep_idx = post_nms_keep_idx
            # self.post_nms_scores = post_nms_scores
            # self.post_nms_topk_keep_idx = post_nms_topk_keep_idx
            # self.detection_per_batch = detection_per_batch
    
        return detections

        
    
    def get_detections(self):
        return self.detections
 
    def debug_outputs(self):
        return (self.class_ids,
                self.indices,
                self.mesh,
                self.ixs,
                self.class_scores,
                self.bbox_delta,
                self.refined_proposals,
                self.clipped_proposals_list,
                self.pre_nms_class_ids_list,
                self.pre_nms_scores_list,
                self.pre_nms_proposals_list,
                )





def debug(proposals=[], mrcnn_class_probs=[], mrcnn_bbox=[], image_window=[], image_shape=[]):
    from MaskRCNN.config import config as conf
    np.random.seed(863)
    
    num_batches = 1
    if len(image_window) == 0:
        image_window = np.array([[131, 0, 893, 1024]], dtype='int32')
    
    if len(image_shape) == 0:
        image_shape = [1024, 1024, 3]
    
    
    if len(proposals) == 0:
        proposals = np.array(np.random.random((1 ,8 ,4)), dtype='float32')
        num_batches = len(proposals)
        
    if len(mrcnn_class_probs) == 0:
        mrcnn_class_probs = np.array(np.random.random((1 ,8 ,4)), dtype='float32')  # [num_batch, num_top_proposal,
    # num_classes]
    
    if len(mrcnn_bbox) == 0:
        mrcnn_bbox = np.array(np.random.random((1 ,8 ,4 ,4)), dtype='float32')

    print('mrcnn_class_probs ', mrcnn_class_probs.shape, mrcnn_class_probs)
    print ('')
    print('mrcnn_bbox ', mrcnn_bbox)
    print('')
    
    print (image_window)
    print (image_shape)
    
    obj_D = DetectionLayer(conf, image_shape, num_batches, image_window, proposals, mrcnn_class_probs, mrcnn_bbox, DEBUG=True)
    (class_ids, indices, mesh, ixs, class_scores, bbox_delta, refined_proposals, clipped_proposals_list, pre_nms_class_ids_list, pre_nms_scores_list, pre_nms_proposals_list )= obj_D.debug_outputs()
    
    detections = obj_D.get_detections()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        (class_ids_, indices_, mesh_, ixs_, class_scores_, bbox_delta_, refined_proposals_, pre_nms_class_ids_list_, pre_nms_scores_list_, pre_nms_proposals_list_, detections_)=  sess.run(
                [class_ids, indices, mesh, ixs, class_scores, bbox_delta, refined_proposals, pre_nms_class_ids_list, pre_nms_scores_list, pre_nms_proposals_list, detections]
        )


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
        print( 'bbox_delta_ ',bbox_delta_.shape, bbox_delta_)
        print('')
        print('refined_proposals_ ', refined_proposals_.shape, refined_proposals_)
        print('')
        print('pre_nms_class_ids_list_ ', len(pre_nms_class_ids_list_), pre_nms_class_ids_list_)
        print('')
        print('pre_nms_scores_list_ ', len(pre_nms_scores_list_), pre_nms_scores_list_)
        print('')
        print('pre_nms_proposals_list_ ', len(pre_nms_proposals_list_), pre_nms_proposals_list_)
        print('')
        print('detections_ ', detections_.shape, detections_)



# debug()
