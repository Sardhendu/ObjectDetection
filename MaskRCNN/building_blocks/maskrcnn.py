'''

1. We have features map in different shapes ranging from 32 to 256 dim
2. This bring the problem of different scales, regardless of the scale we should be assign the anchor box to the right location in the image. Here we take care of that.
'''

from MaskRCNN.building_blocks import ops
import tensorflow as tf
import numpy as np
import keras.backend as K
import keras.layers as KL






class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)
    

class MaskRCNN():
    def __init__(self, image_shape, pool_shape, num_classes, levels, proposals, feature_maps, type='keras',
                 DEBUG=False):
        self.image_shape = image_shape
        self.pool_shape = pool_shape
        self.num_classes = num_classes
        self.levels = levels
        self.DEBUG = DEBUG
        
        self.build(proposals, feature_maps, type)
    
    def build(self, proposals, feature_maps, type='keras', DEBUG=False):
        self.roi_pooling(self.image_shape, self.pool_shape, self.levels, proposals, feature_maps)
        
        if type=='tf':
            self.classifier_with_fpn_tf()
        else:
            self.classifier_with_fpn_keras()

    def log2_graph(self, x):
        """Implementatin of Log2. TF doesn't have a native implemenation."""
        return tf.log(x) / tf.log(2.0)

    def roi_pooling(self, image_shape, pool_shape, levels, proposals, feature_maps):
        '''
        My Notes:

        While we did the proposal generation, we selected proposals by stacking anchors from each feature map.
        The problem is that, in doing so, we were able to get the best possible proposal, but at the same time we lost
        track of which proposal in a batch were part of which feature map. Now for ROI pooling (to perform crop and
        resize operation) we need to know the proposals and the feature map they were generated from. Here we take an
        attempt to first know which proposals were generated from which feature map and then we crop/resize a 7x7
        regions
        from those feature map for object detection and bounding box refinement.

        This preprocess is very easy when we use a single Resnet, VGG or any other network because we just generate a
        single feature map of one scale.

        Important Variables:

        1. pooled_rois =
            Initial before sorting: [proposals*num_batch, 7, 7, 256]. This is the pooled region that goes to the object
            detection and bounding box refinement module.
            After = [num_batch, proposals, 7, 7, 256], where proposals are sorted based on which feature map they are
            obtained from
        2. box_to_level: [proposals*num_batches, 2], where 2 represents [batch_num, image_num]. Why is this variable
        important. The pooled_rois after the for loop are sorted in the order of feature map. i.e. proposals from
        one feature map are grouped together. But our input is the terms if [batch_num, num_proposals, 4]. Inorder to
        identify the ROI instance (i.e the batch and which proposal the roi was extracted) we keep box_to_level,
        which serves
        as the identifying ID(key).

        Note: Sorting is done with batch_num as 1st priority followed by proposal generated from sorted feature_map as
        second priority
        '''
        k0 = 4
        min_k = min(levels)
        max_k = max(levels)
        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(proposals, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
    
        # Use shape of first image. Images in a batch must have the same size.
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = self.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = k0 + tf.cast(tf.round(roi_level), tf.int32)
        roi_level = tf.minimum(max_k, tf.maximum(min_k, roi_level))
        roi_level = tf.squeeze(roi_level, 2)
    
        # # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled_rois = []
        box_to_level = []  # Provides
        for i, level in enumerate(levels):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(proposals, ix)
        
            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)
        
            # Keep track of which box is mapped to which level
            box_to_level.append(ix)
        
            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)
            
            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
        
            # Basically this says the bounding box to crop from the image(feature_map) and then
            pooled_rois.append(tf.image.crop_and_resize(
                    feature_maps[i], level_boxes, box_indices, pool_shape,
                    method="bilinear"))
    
        pooled_rois = tf.concat(pooled_rois, axis=0)  # [proposals*num_batch, 7, 7, 256]
        box_to_level = tf.concat(box_to_level, axis=0)
    
        # We do the below just to sort the roi_pools based on the batch_num and anchor_num. Because tensorflow
        # doesn't have an implicit function for that we take a d-tour
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
    
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)
    
        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index, TF doesn't have a way to sort by two columns, so merge them and
        # sort. Assumption: We will never have more than 100000 combination of batch and anchors
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        # Get the indices from the [::-1] column (column added in box_range) using the sorting tensor and then sort the
        # pooled_rois
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        # ix = tf.gather(box_to_level[:, 2], ix)  # This is redundant, but still we keep it
        pooled_rois = tf.gather(pooled_rois, ix)
        
        if self.DEBUG:
            self.box_to_level = box_to_level
            self.sorting_tensor = sorting_tensor
            self.ix = ix
        else:
            self.box_to_level = []
            self.sorting_tensor = []
            self.ix = []
    
        # Add the batch Dimension
        self.pooled_rois = tf.expand_dims(pooled_rois, 0)
        # return pooled_rois
        
    def classifier_with_fpn_tf(self):
        print (self.pooled_rois.shape)

        rois_shape = self.pooled_rois.get_shape().as_list()
        
        # Note we dont perform batch normalization, because as per matterport github implementation of Mask RCNN,
        # it is suggested to not have it becasue batch norm doesnt perform well with very small batches.
        # FC Layer 1
        x = tf.concat([
                tf.stack([
                    ops.activation(ops.conv_layer(self.pooled_rois[i], k_shape=self.pool_shape + [rois_shape[-1], 1024],
                                   stride=1, padding='VALID', scope_name='mrcnn_class_conv1'), 'relu', 'FC1_relu')
                    for i in range(0,rois_shape[0])])],
                axis=0)
        self.FC1 = x if self.DEBUG else []

        # FC Layer 2
        x = tf.concat([
                tf.stack([
                    ops.activation(ops.conv_layer(x[i], k_shape=[1,1,1024,1024],
                                  stride=1, padding='VALID', scope_name='mrcnn_class_conv2'), 'relu', 'FC2_relu')
                    for i in range(0, rois_shape[0])])],
                    axis=0)
        self.FC2 = x if self.DEBUG else []
        
        
        # Squeeze the 2nd and 3rd dimension [num_batch, num_proposals, 1, 1, 1024] to [num_batch, num_proposals, 1024]
        shared = tf.squeeze(x,[2,3])
        self.shared = shared if self.DEBUG else []
        
        with tf.variable_scope('mrcnn_class_scores'):
            mrcnn_class_logits = tf.concat([
                tf.stack([
                    ops.fc_layers(shared[i], k_shape=[1024, self.num_classes],scope_name='mrcnn_class_logits')
                    for i in range(0, rois_shape[0])])],
                    axis=0)

            self.mrcnn_probs = tf.concat([
                tf.stack([
                    ops.activation(mrcnn_class_logits[i], 'softmax', scope_name='mrcnn_class')
                    for i in range(0, rois_shape[0])])],
                    axis=0)
            
        with tf.variable_scope('mrcnn_class_bbox'):
            x = tf.concat([
                tf.stack([
                    ops.fc_layers(shared[i], k_shape=[1024, self.num_classes*4], scope_name='mrcnn_bbox')
                    for i in range(0, rois_shape[0])])],
                    axis=0)

            s = tf.shape(x)
            self.mrcnn_bbox = tf.reshape(x, [s[0], s[1], self.num_classes, 4], name="mrcnn_bbox")


    def classifier_with_fpn_keras(self):
        '''
        Detecting Objects and Refining Bbox:
        
        The paper says to have two fully connected layer, In theory, there is no such thing as fully connected layer
        with convolution based architecture, but there are 1x1 convolutions.
        
        TimeDistributed: The output from the ROI pooling is [numb_batches, num_proposals, 7, 7, 4],
        Normally convolution operates on [num_batches, 7, 7, 4]. In order to make convolution operation work on each
        proposals per batch we use the Keras Time Distributed method.
        
        This can also be achieved by simply loop anc concat tensor operation or Flatten, perform convolution and
        reshape.

        '''
        
        # FC_CONV layer 1: We perform a 7x7 convolution to create x=[batch_like_num, 1, 1, 1024] , This is equivalent
        # to a FC layer but with convolutional style
        print (self.pooled_rois.shape)
        x = KL.TimeDistributed(KL.Conv2D(1024, (self.pool_shape[0], self.pool_shape[1]), padding="valid"),
                               name='mrcnn_class_conv1')(self.pooled_rois)
        x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=False)
        x = KL.Activation('relu')(x)
        self.FC1 = x if self.DEBUG else []

        # FC_CONV layer 2: Perform 1x1 convolutions
        x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)), name="mrcnn_class_conv2")(x)
        x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=False)
        x = KL.Activation('relu')(x)
        self.FC2 = x if self.DEBUG else []

        # Shared Convolution across the Classifier and Regressor
        shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)
        self.shared = shared if self.DEBUG else []

        # Classifier (Object detection)
        with tf.variable_scope('mrcnn_class_scores'):
            mrcnn_class_logits = KL.TimeDistributed(KL.Dense(self.num_classes),
                                                    name='mrcnn_class_logits')(shared)
            self.mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                             name="mrcnn_class")(mrcnn_class_logits)
            
        # Bbox Refiner
        with tf.variable_scope('mrcnn_bbox'):
            x = KL.TimeDistributed(KL.Dense(self.num_classes * 4, activation='linear'),
                                   name='mrcnn_bbox_fc')(shared)
            # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
            s = K.shape(x)
            self.mrcnn_bbox = KL.Reshape((s[1], self.num_classes, 4), name="mrcnn_bbox")(x)
    
    
    def get_rois(self):
        return self.pooled_rois
    
    def get_mrcnn_class_probs(self):
        return self.mrcnn_probs
    
    def get_mrcnn_bbox(self):
        return self.mrcnn_bbox
    
    def debug_outputs(self):
        return self.box_to_level, self.sorting_tensor, self.ix, self.FC1, self.FC2, self.shared

    
    
    
    


def debugg():
    from MaskRCNN.config import config as conf
    from MaskRCNN.building_blocks.proposals_tf import Proposals

    np.random.seed(255)
    num_batches = 2

    # P2 = (2, 256, 256, 256), P3 = (2, 128, 128, 256), P4 = (2, 64, 64, 256), P5 = (2, 32, 32, 256)

    P2 = np.array(np.random.random((num_batches, 256, 256,256)), dtype='float32')
    P3 = np.array(np.random.random((num_batches, 128, 128, 256)), dtype='float32')
    P4 = np.array(np.random.random((num_batches, 64, 64, 256)), dtype='float32')
    P5 = np.array(np.random.random((num_batches, 32, 32, 256)), dtype='float32')
    feature_maps = [P2, P3, P4, P5]

    a = np.array(np.random.random((2, 6, 2)), dtype='float32')
    b = np.array(np.random.random((2, 6, 4)), dtype='float32')
    c = np.array(np.random.random((2, 6, 4)), dtype='float32')


    # Proposal Layer
    obj_p = Proposals(conf, batch_size=num_batches)
    p_graph = obj_p.get_proposal_graph()

    # Mask RCNN : ROI Pooling
    obj_MRCNN = MaskRCNN(image_shape=[800, 1024], pool_shape=[7, 7], num_classes=81, levels=[2, 3, 4, 5],
                         proposals=p_graph['proposals'], feature_maps=feature_maps, type='keras', DEBUG=True)
    pooled_rois = obj_MRCNN.get_rois()
    mrcnn_class_probs = obj_MRCNN.get_mrcnn_class_probs()
    mrcnn_bbox = obj_MRCNN.get_mrcnn_bbox()
    box_to_level, sorting_tensor, ix, FC1, FC2, shared = obj_MRCNN.debug_outputs()

    feed_dict = {p_graph['rpn_probs']: a, p_graph['rpn_bbox']: b, p_graph['input_anchors']: c}
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        (proposals_, pooled_rois_, box_to_level_, sorting_tensor_, ix_, FC1_, FC2_, shared_, mrcnn_class_probs_,
         mrcnn_bbox_)  =  sess.run([p_graph['proposals'], pooled_rois, box_to_level, sorting_tensor, ix, FC1, FC2,
                                    shared, mrcnn_class_probs, mrcnn_bbox], feed_dict=feed_dict)

        print ('')
        print('TENSORFLOW STUFF.........')
        print('proposals ', proposals_)
        print('')
        print ('box_to_level ', box_to_level_)
        print ('')
        print('sorting_tensor ', sorting_tensor_)
        print('')
        print ('box_indice ', ix_)
        print('')
        print('pooled ', pooled_rois[0].shape)
        print('')
        print('')
        print ('FC1_ shape: ', FC1_.shape, FC1_)
        print('')
        print('FC2_ shape: ', FC2_.shape, FC2_)
        print ('')
        print('shared shape: ', shared_.shape, shared_)
        print('mrcnn_class_probs_ ', mrcnn_class_probs_.shape, mrcnn_class_probs_)
        print('')
        print('mrcnn_bbox_ ', mrcnn_bbox_.shape, mrcnn_bbox_)
        print('')
    #

# debugg()




