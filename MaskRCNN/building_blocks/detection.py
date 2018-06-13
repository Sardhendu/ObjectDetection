

import numpy as np
import tensorflow as tf



class DetectionLayer():
    def __init__(self, mrcnn_class_probs, mrcnn_bbox, DEBUG=False):
        self.DEBUG = DEBUG
        self.build(mrcnn_class_probs, mrcnn_bbox)
    
    def build(self, mrcnn_class_probs, mrcnn_bbox):
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
        bbox  = tf.gather_nd(mrcnn_bbox, ixs)

        # # Apply Box Delta
        # apply_box_deltas(bbox)
        
        if self.DEBUG:
            self.class_ids = class_ids
            self.indices = indices
            self.mesh = mesh
            self.ixs = ixs
            self.class_scores = class_scores
            self.bbox = bbox
        # else:
            # self.class_ids = []
        # self.indices = []
        # self.mesh = []
        # self.class_scores = []
        # self.bbox = []

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
        normalize anchors that have un-normalized coordinate values). After this we convert the pre_nms_anchors back
        to the
        original shape of [num_batches, num_anchors, (y1, x1,y2, x2)]

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
    
    def debug_outputs(self):
        return self.class_ids, self.indices, self.mesh, self.ixs, self.class_scores, self.bbox
    
    



def debug():
    np.random.seed(863)
    mrcnn_class_probs = np.random.random((2,3,4)) #[num_batch, num_top_proposal, num_classes]
    mrcnn_bbox = np.random.random((2,3,4,4))
    
    print('mrcnn_class_probs ', mrcnn_class_probs.shape, mrcnn_class_probs)
    print ('')
    print('mrcnn_bbox ', mrcnn_bbox)
    print('')
    
    
    class_ids, indices, mesh, ixs, class_scores, bbox = DetectionLayer(mrcnn_class_probs, mrcnn_bbox,
                                                               DEBUG=True).debug_outputs()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        class_ids_, indices_, mesh_, ixs_, class_scores_, bbox_ = sess.run(
                [class_ids, indices, mesh, ixs, class_scores, bbox])#,
        # class_scores,
        #  bbox])
        print(class_ids_.shape, class_ids_)
        print('ind ', indices_)
        print (mesh_)
        print(ixs_)
        print(class_scores_.shape, class_scores_)
        print(bbox_.shape, bbox_)

debug()
    
    