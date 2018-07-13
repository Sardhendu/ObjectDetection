

import tensorflow as tf
import keras.backend as K

class Loss():
    
    def __init__(self):
        pass
    
    @staticmethod
    def rpn_class_loss(rpn_target_class, rpn_class_logits):
        '''
        :param rpn_target_class:  target class [batch_size, num_anchors, 1], here -1 = negative_class(iou<0.3), +1 = positive_class (iou>0.7), 0 = neutral
        :param rpn_class_probs:   predicted class [batch_size, num_anchors, (fg_probs, bg_probs)]
        
        The idea is to only gather data for -1 and +1 and avoid neutrals (0)
        :return: loss (scalar)
        
        '''

        # Convert rpn_target_class to [batch_size, num_anchors] with logits values
        rpn_target_class = tf.squeeze(rpn_target_class, axis=-1)
        print (rpn_target_class.shape)

        # Select +1 class and -1 class, since 0 neutral doesnt contribute to loss
        indices = tf.where(tf.not_equal(rpn_target_class, 0))
        target_class = tf.gather_nd(rpn_target_class, indices)

        # Sigmoid cross entropy requires classes 1 and 0, we change -1 class to 0
        target_class = tf.cast(tf.equal(target_class, 1), tf.int32)
        
        # Select the prediction using the indices
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        
        # Now that we have the rarget class and logits, we will compute cross entropy loss
        loss = K.sparse_categorical_crossentropy(target=target_class,
                                                 output=rpn_class_logits,
                                                 from_logits=True)

        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        
        return loss
    
    
    @staticmethod
    def rpn_box_loss(rpn_target_bbox, rpn_pred_box, rpn_target_class, batch_size):
        '''
        :param rpn_target_class:  [batch_size, num_anchors, 1] # required becasue we need to capture only +ve classes
        :param rpn_target_bbox:    [batch_size, max_gt_boxes(100), (c_y, c_x, log(dh), log(dw)]
        :param rpn_pred_box:      [batch_size, anchors, (c_y, c_x, log(dh). log(dw)]
        :return:
        
        Note: rpn_target_bbox are zero-padded, they were added to fulfil the total number of box and concat for each objects. we assume that given an image we would at most find 100 objects.
        
        Losses are to be calculated on only Foreground, the rpn_target_bbox corresponds to only +ve classes,
        '''
        rpn_target_class = tf.squeeze(rpn_target_class, -1)
        
        # rpn_pre_box = [batch_size, 100, 4] here only few out of 100 box would be +ve classes rest all are zero padded. Inorder to compare them we find the boxes from rpn_pred_box corresponding to +ve class.
        indices = tf.where(tf.equal(rpn_target_class, 1))
        rpn_pred_box_pos = tf.gather_nd(rpn_pred_box, indices)
        
        # Gather from rpn_target_bbox where teh values are not zero. Basically the top "n" boxes are non zero. Also Note this has to be done for every batch
        # Get data for count of non-padded (non=zero) records for each batch
        non_pad_count = tf.reduce_sum(tf.cast(tf.equal(rpn_target_class, 1), tf.int32), axis=1)# K.sum(K.cast(K.equal(rpn_target_class, 1), tf.int32), axis=1)#
        rpn_target_bbox_nopad = []
        for i in range(0,batch_size):
            rpn_target_bbox_nopad.append(rpn_target_bbox[i,:non_pad_count[i]]) # non_pad_count[i] The count of non-zeros records in batch i
        rpn_target_bbox_nopad = tf.concat(rpn_target_bbox_nopad, axis=0)
        #
        # # Now that we have two boxes of same size, lets get the Regression loss.
        # # L1=smooth norm  =     | 0.5(x_sq)   if |x| < 1
        # #                       | |x| - 0.5   otherwise
        l1_dist = tf.abs(rpn_target_bbox_nopad - rpn_pred_box_pos)
        less_than_one = tf.cast(tf.less(l1_dist, 1.0), tf.float32)
        loss = (0.5*less_than_one * l1_dist**2) + ((l1_dist-0.5) * (1-less_than_one))

        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        
        # TODO: The problem with random weights, the loss becomes NaN because the prediction box are very high negative numbers. try using pretrained weights, and see if the problem vanishes.
        return rpn_pred_box_pos, loss

    @staticmethod
    def mrcnn_class_loss(mrcnn_target_class, mrcnn_pred_logits, batch_active_class_ids, sess):
        '''

        :param mrcnn_target_class: Zero padded [batch_size, 100,4],
                                    4    -> number of classes
                                    100  -> maximum number of objects to be detected
        :param mrcnn_class_logits:      [batch_size, 100, 4]
                                    100 -> Number of rois
        :param batch_active_class_ids:  [batch_size, 4]
                            # Has a value of 1 for each object in the image of a batch
        :return:

        '''
        mrcnn_target_class = tf.cast(mrcnn_target_class, dtype=tf.int64)
        batch_active_class_ids = tf.cast(batch_active_class_ids, dtype=tf.int32)
        mrcnn_pred_logits = tf.cast(mrcnn_pred_logits, dtype=tf.float32)
    
        pred_class_ids = tf.argmax(mrcnn_pred_logits, axis=2)
    
        # Perform the cross entropy loss,
        # Note mrcnn_target_class, mrcnn_pred_logits are not one-hot-coded, they are in labe;
        
        # loss = tf.losses.sparse_softmax_cross_entropy(mrcnn_target_class, pred_class_ids)
    
        t = sess.run(mrcnn_target_class)
        l = sess.run(mrcnn_pred_logits)
        p = sess.run(pred_class_ids)
        print('mrcnn_target_class ', t)
        print('')
        print('mrcnn_class_logits ', l)
        print('')
        print('batch_active_class_ids ', sess.run(batch_active_class_ids))
        print('')
        print('pred_class_ids ', )
        # print('')
        # print('loss ', sess.run(loss))
        
        print (t.shape, l.shape, p.shape)
        
        
        
        # TODO: Somehow the mrcnn module return 1000 bbox and labels, this should be only 100 check why?


        
        
        
        