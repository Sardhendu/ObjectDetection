

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
        :param rpn_target_bbox:    [batch_size, max_gt_boxes(100), (c_dy, c_dx, log(dh), log(dw)]
        :param rpn_pred_box:      [batch_size, anchors, (c_dy, c_dx, log(dh). log(dw)]
        :param rpn_target_class:  [batch_size, num_anchors, 1] # required becasue we need to capture only +ve classes
        :return:
        
        Note: rpn_target_bbox are zero-padded, they were added to fulfil the total number of box and concat for each objects. we assume that given an image we would at most find 100 objects.
        
        Losses are to be calculated on only Foreground, the rpn_target_bbox corresponds to only +ve classes,
        '''
        rpn_target_class = tf.squeeze(rpn_target_class, -1)
        
        # rpn_pred_box = [batch_size, 100, 4] here only few out of 100 box would be +ve classes rest all are zero padded. Inorder to compute loss b/w rpn_target_bbox and rpn_pred_box we find the boxes from rpn_pred_box corresponding to +ve class.
        indices = tf.where(tf.equal(rpn_target_class, 1))
        rpn_pred_box_pos = tf.gather_nd(rpn_pred_box, indices)
        
        # Gather from rpn_target_bbox where the values are not zero. Basically the top "n" boxes are non zero. Also Note this has to be done for every batch
        # Get data for count of non-padded (non=zero) records for each batch
        non_pad_count = tf.reduce_sum(tf.cast(tf.equal(rpn_target_class, 1), tf.int32), axis=1)# K.sum(K.cast(K.equal(rpn_target_class, 1), tf.int32), axis=1)#
        rpn_target_bbox_nopad = []
        for i in range(0,batch_size):
            rpn_target_bbox_nopad.append(rpn_target_bbox[i,:non_pad_count[i]]) # non_pad_count[i] The count of non-zeros records in batch i
        rpn_target_bbox_nopad = tf.concat(rpn_target_bbox_nopad, axis=0)

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
    def mrcnn_class_loss(mrcnn_target_class_ids, mrcnn_pred_logits, batch_active_class_ids):
        '''
        :param mrcnn_target_class_ids: Zero padded [batch_size, MRCNN_TRAIN_ROIS_PER_IMAGE],
                                    4    -> number of classes
                                    100  -> maximum number of objects to be detected
        :param mrcnn_pred_logits:      [batch_size, MRCNN_TRAIN_ROIS_PER_IMAGE, num_objects]
                                    32 -> Number of rois specifically for the shape data set
        :param batch_active_class_ids:  [batch_size, num_objects]
                            # Has a value of 1 for each object in the image of a batch
        :return:
        '''
        
        
        # Inorder to compute loss we need to convert mrcnn_target_class into
        # [batch_size, MRCNN_TRAIN_ROIS_PER_IMAGE, num_objects] (one hot vectors per class) and use "softmax cross
        # entropy", OR we could convert mrcnn_pred_logits into [batch_size, MRCNN_TRAIN_ROIS_PER_IMAGE] and do
        # "sparse_softmax_ross_entropy". Later is easy so let do that
        
        mrcnn_target_class_ids = tf.cast(mrcnn_target_class_ids, dtype='int64')
        mrcnn_pred_logits = tf.cast(mrcnn_pred_logits, dtype='float32')
        batch_active_class_ids = tf.cast(batch_active_class_ids, dtype='float32')
        

        mrcnn_pred_class_ids = tf.argmax(mrcnn_pred_logits, axis=2)

        pred_active = tf.gather(batch_active_class_ids[0], mrcnn_pred_class_ids)
        
        
        # Perform the cross entropy loss,
        # Note mrcnn_target_class, mrcnn_pred_logits are not one-hot-coded, they are label-coded
        # There would be no loss for zeros-paded data
        
        # t = sess.run(mrcnn_target_class_ids)
        # # l = sess.run(mrcnn_pred_logits)
        # p = sess.run(mrcnn_pred_logits)
        # a = sess.run(batch_active_class_ids)

        # print(t.shape, t)
        # print('')
        # print(l.shape, l)
        # print('')
        # print(p.shape, p)
        # print('')
        # print(a)
        # print(pred_active)


        # THe below may seem weird becasue mrcnn_target_class_ids = [batch_size, N] and
        # mrcnn_pred_logits = [batch_size, N, num_objects] (they are different dimension). But this is way
        # tf.nn.sparse_softmax_cross_entropy_with_logits expects its inputs
        
        # TODO: Why the Loss turns out to be nan, even when we have o object in the image. basically pred_active > 0
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=mrcnn_target_class_ids,
                logits=mrcnn_pred_logits
        )

        # Remove losses of predictions of classes that are not in the active
        # classes of the image.
        loss = loss * pred_active
        
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
        return pred_active, loss


    @staticmethod
    def mrcnn_box_loss(mrcnn_target_box, mrcnn_pred_box, mrcnn_target_class_ids, batch_size=2):
        '''
        
        :param mrcnn_target_box:    zero padded [batch_size, num_rois , (cy, cx, log(h), log(w)]  -> cx, cy normalized
        :param mrcnn_pred_box:
        :return:
        '''
        import numpy as np
        print('mrcnn_target_box %s \n'%str(mrcnn_target_box.shape), mrcnn_target_box)
        print('')
        print ('mrcnn_pred_box %s \n'%str(mrcnn_pred_box.shape), mrcnn_pred_box)
        print('')
        print('mrcnn_target_class_ids %s \n'%str(mrcnn_target_class_ids.shape), mrcnn_target_class_ids)

        target_boxes = []
        pred_boxes = []
        for i in range(0, batch_size):
            positive_roi_ix = tf.where(mrcnn_target_class_ids[i] > 0)[:, 0]
            positive_roi_class_ids = tf.cast(
                    tf.gather(mrcnn_target_class_ids[i], positive_roi_ix), tf.int64)
            
            # Get mrcnn_target_box for indices
            target_box = tf.gather(mrcnn_target_box[i], positive_roi_ix)
            
            # Get the mrcnn_pred_box for the indices and especially for the mrcnn_target_class_ids
            pred_box = tf.gather(mrcnn_pred_box[i], positive_roi_ix)
            indices = tf.stack([tf.cast(tf.range(start=0,
                                                 limit=tf.shape(positive_roi_class_ids)[0]), dtype=tf.int64),
                                positive_roi_class_ids], axis=1)

            pred_bbox1 = tf.gather_nd(pred_box, indices)

            # TODO: Stack target_boxes and pred_bbox1 one upon another
            # Select the prediction boxes
            target_boxes.append(target_box)
            pred_boxes.append(pred_bbox1)

        target_boxes = tf.concat(target_boxes, axis=0)
        pred_boxes = tf.concat(pred_boxes, axis=0)

        # Only positive ROIs contribute to the Loss
        # Compute binary cross entropy. If no positive ROIs, then return 0.
        # shape: [batch, roi, num_classes]
        loss = K.switch(tf.size(target_boxes) > 0,
                        K.binary_crossentropy(target=target_boxes, output=pred_boxes),
                        tf.constant(0.0))
        loss = K.mean(loss)
        
        # TODO: Calculate the loss between stacked target_boxes and stacked pred_boxes1
        return loss







def debug():
    import numpy as np
    
    mrcnn_target_box = np.random.random((2,32,4))
    mrcnn_pred_box = np.random.random((2,32,4,4))
    mrcnn_target_class_ids = np.zeros((2, 32))
    mrcnn_target_class_ids[0,[2]] = 1
    mrcnn_target_class_ids[0,[3]] = 2
    mrcnn_target_class_ids[1,[4]] = 1
    
    print ('mrcnn_target_box ', mrcnn_target_box[0])
    print('')
    print('mrcnn_pred_box ', mrcnn_pred_box[0])
    print('')
    print('mrcnn_target_class_ids ', mrcnn_target_class_ids[0])
    
    
    
    a = tf.placeholder(dtype=tf.float32, shape=(2,32,4), name='a')
    b = tf.placeholder(dtype=tf.float32, shape=(2,32,4, 4), name='b')
    c = tf.placeholder(dtype=tf.int32, shape=(2,32), name='c')
    
    
    with tf.Session() as sess:
        positive_roi_ix, target_boxes, pred_bbox, indices, pred_bbox1, pred_bbox2, loss = Loss.mrcnn_box_loss(a, b, c)
        
        sess.run(tf.global_variables_initializer())
    
        positive_roi_ix_, target_boxes_, pred_bbox_, indices_, pred_bbox1_, pred_bbox1__, loss_ = sess.run([
            positive_roi_ix, target_boxes, pred_bbox, indices, pred_bbox1, pred_bbox2, loss],
                feed_dict={a:mrcnn_target_box, b:mrcnn_pred_box, c:mrcnn_target_class_ids})
        print('')
        print('')
        print(positive_roi_ix_)
        print('')
        print('')
        print(target_boxes_)
        print('')
        print('')
        print(pred_bbox_)
        print('')
        print('')
        print(indices_)
        print('')
        print('')
        print(pred_bbox1_)
        print('')
        print('')
        print(pred_bbox1__)
        print('')
        print('')
        print(loss_)
        
        