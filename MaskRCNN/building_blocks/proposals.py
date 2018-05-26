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




def nms():
    '''
    nms = Non Max Suppression
    :return:
    '''

def clip_boxes_to_01(box_delta, window=[0.0,0.0,1.0,1.0]):
    '''
    :param box_delta: The anchor per pixel position boxes for each batch with 4 pixel coordinates.
    :param window: THe min and max coordinates of window (We use this because our predictions should lie i 0,1 range)
    :return:
    
    The idea is pretty basic here:
        1. We split the coordinates.
        2. Check if they lie in the window range, if not make them lie
        3. Then concat them back to the original shape
    More over bring the box coordinate prediction to the range of [0,1] also helps us performing the next step i.e
    non-max suppression
    '''
    # Window: [0,0,1,1] # 0,0 represents the top left corner and 1,1 represents the bottom right corner
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(box_delta, 4, axis=2)

    before_xy = [y1, x1, y2, x2]
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    
    after_xy = [y1, x1, y2, x2]
    clipped = tf.concat([y1, x1, y2, x2], axis=2, name="clipped_boxes")

    return before_xy, after_xy, clipped


def gather_data_for_idx(ix, foreground_probs, box_delta, anchors):
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
    mesh = tf.meshgrid(tf.range(tf.shape(ix)[1]), tf.range(tf.shape(ix)[0]))[1]
    ixs = tf.stack([mesh, ix], axis=2)

    # Gather only the data pertaining to the ixs
    foreground_probs = tf.gather_nd(foreground_probs, ixs)
    logging.info('foreground_probs shape = %s', str(foreground_probs.shape))
    
    # Gather only the data pertaining to the ixs
    box_delta = tf.gather_nd(box_delta, ixs)
    logging.info('Box delta shape = %s', str(box_delta.shape))

    # Gather only the data pertaining to the ixs
    anchors = tf.gather_nd(anchors, ixs)
    logging.info('anchors shape = %s', str(anchors.shape))
    
    return ixs, mesh, foreground_probs, box_delta, anchors


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
    foreground_probs = inputs[0][:,:,1]
    logging.info('Foreground_probs shape: %s', str(foreground_probs.get_shape().as_list()))
    
    # Box deltas = [batch, num_rois, 4]
    box_delta = inputs[1] * np.reshape(conf.RPN_BBOX_STD_DEV, [1, 1, 4])
    logging.info('box_delta shape: %s', str(box_delta.get_shape().as_list()))
    
    # Get the anchors [None, 2]
    anchors = inputs[2]
    logging.info('anchors shape: %s', str(anchors.get_shape().as_list()))
    
    # Searching through lots of anchors can be time consuming. So we would select at most 6000 of them for further processing
    max_anc_before_nms = tf.minimum(3, tf.shape(anchors)[1])
    logging.info('max_anc_before_nms shape: %s', str(max_anc_before_nms))
    
    # Here we fetch the idx of the top 6000 anchors
    ix = tf.nn.top_k(foreground_probs, max_anc_before_nms, sorted=True, name="top_anchors").indices
    logging.info('ix shape: %s', str(ix.get_shape().as_list()))
    
    # Now that we have the idx of the top anchors we would want to only gather the data related pertaining to those idx. We would wanna gather foreground_prob and box_delta only for the selected anchors.
    # foreground_probs = tf.gather_nd(foreground_probs, ix)
    ixs, mesh, foreground_probs, box_delta, anchors = gather_data_for_idx(ix, foreground_probs, box_delta, anchors)
    
    # The box_delta can have values at the interval of 0,1
    window = np.array([0, 0, 1, 1], dtype=np.float32)
    before_xy, after_xy, clipped = clip_boxes_to_01(box_delta, window=window)
    
    # Perform Non-max supppression.
    

    outs = [foreground_probs, box_delta, anchors, max_anc_before_nms, ix, ixs, mesh, before_xy, after_xy, clipped]
    
    return outs
    
def debugg():
    from MaskRCNN.config import config as conf

    rpn_probs = tf.placeholder(dtype=tf.float32,
                               shape=[None, None, 2],
                               name="rpn_prob")

    rpn_box = tf.placeholder(dtype=tf.float32,
                             shape=[None, None, 4],
                             name="rpn_box")

    input_anchors = tf.placeholder(dtype=tf.float32,
                                   shape=[None, None, 4],
                                   name="input_anchors")
    
    outs = proposals(conf, inputs=[rpn_probs, rpn_box, input_anchors])
    foreground_probs, box_delta, anchors, max_anc_before_nms, ix, ixs, mesh, before_xy, after_xy, clipped = outs
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        a= np.random.random((2, 5, 2))
        b= np.random.random((2, 5, 4))
        c= np.random.random((2, 5, 4))
        feed_dict={rpn_probs: a, rpn_box: b, input_anchors: c}

        print('rpn_probs ', a)
        print('')
        print('rpn_box ', b)
        print('')
        print('anchors ', c)
        run_out = sess.run([foreground_probs, box_delta, anchors, ix, ixs, mesh, before_xy, after_xy, clipped], feed_dict=feed_dict)
        fp, bxd, anch, ix_, ixs_, mesh_, before_xy_, after_xy_, clipped_ = run_out
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
        
        print ('')
        print ('......................... ')
        for bfi in before_xy_:
            # print (bfi)
            print('before_xy_ shape = %s, %s'%(str(bfi.shape), str(bfi)))
            # print(np.maximum(np.minimum(bfi, 1), 0))
            print('')
            
        print('')
        print('')
        for bfi in after_xy_:
            # print (bfi)
            print('after_xy_ shape = %s, %s'%(str(bfi.shape), str(bfi)))
            # print(np.maximum(np.minimum(bfi, 1), 0))
            print('')
            
        print('')
        print('')
        print('clipped_ shape = %s , %s'%(str(clipped_.shape), str(clipped_)))
        # print('')
        # print('x2_ = ', x2_)

        # print('')
        # print('fp2 = ', fp2)


debugg()

# def get_slices_given_idx(inputs, batch_size):
#     for i in range(0,batch_size):
#         inputs_slice = [x[i] for x in inputs]


# np.random.seed(675)
# a = tf.placeholder(dtype=tf.float32, shape=(2,4,2), name='a')
# a_fore = a[:,:,1]
#
# a_delta = tf.placeholder(dtype=tf.float32, shape=(2,4,4), name='a_del')
# # top = tf.nn.top_k(a_fore, 3, sorted=True, name="top_anchors")
# ix = tf.nn.top_k(a_fore, 3, sorted=True, name="top_anchors").indices
# # po = [[a_fore, ix], lambda x, y: tf.gather_nd(x, y)]
#
# # For Foreground_prob
# hmm = [[[0,2],[0,0]], [[1,2],[1,0]]]
# b = tf.gather_nd(a_fore, hmm)
# b_del = tf.gather_nd(a_delta, hmm)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     aa, a_del, aa_fore, top_, ix_, bb, bb_del= \
#         sess.run([a,a_delta, a_fore,top, ix, b,b_del],
#                  feed_dict={a:np.random.random((2,4,2)),
#                             a_delta:np.random.random((2,4,4))})
#     print(aa.shape, aa)
#     print('')
#     print(aa_fore.shape, aa_fore)
#     print('')
#     # print (top_)
#     # print('')
#     print(ix_.shape, ix_)
#     print('')
#     print(bb.shape, bb)
#
#     print ('')
#     print('')
#     print (a_del.shape, a_del)
#     print('')
#     print(bb_del.shape, bb_del)



#
# i = tf.constant(0)
# while_condition = lambda i: tf.less(i, input_placeholder[1, 1])
# def body(i):
#     # do something here which you want to do in your loop
#     # increment i
#     return [tf.add(i, 1)]
#
# # do the loop:
# r = tf.while_loop(while_condition, body, [i])


# def fun(a):
#     return (a[0] , a[1])
#
#
# my_arg = tf.constant(3, dtype=tf.int64)
# elems = tf.constant([[1,2],[3,4],[5,6]])
# # sum = tf.map_fn(fun, elems, dtype=tf.int32)
# sum = tf.map_fn(lambda x: (x[0], x[1]), (elems, elems), dtype=(tf.int32, tf.int32))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a = sess.run(sum)
#     print (a)










#
#
# np.random.seed(675)
# a = tf.placeholder(dtype=tf.float32, shape=(2,4,2), name='a')
# a_fore = a[:,:,1]
#
# a_delta = tf.placeholder(dtype=tf.float32, shape=(2,4,4), name='a_del')
# # top = tf.nn.top_k(a_fore, 3, sorted=True, name="top_anchors")
# ix = tf.nn.top_k(a_fore, 3, sorted=True, name="top_anchors").indices
#
# #
# mesh = tf.meshgrid(tf.range(ix.shape[1]), tf.range(ix.shape[0]))[1]
# ixs = tf.stack([mesh, ix], axis=2)
#
#
# # For Foreground_prob
# hmm = [[[0,2],[0,0],[0,3]],[[1,2],[1,0],[1,1]]]
# fore = tf.gather_nd(a_fore, hmm)
# fore_nw = tf.gather_nd(a, ixs)[:,:,1]
# b_del = tf.gather_nd(a_delta, ixs)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     aa, aa_fore, a_del, ix_, mesh_, ixs_, b_del_, fore_, fore_nw_= \
#         sess.run([a, a_fore, a_delta, ix, mesh, ixs, b_del, fore, fore_nw],
#                  feed_dict={a:np.random.random((2,4,2)),
#                             a_delta:np.random.random((2,4,4))})
#     print(aa.shape, aa)
#     print('')
#
#     print(aa_fore.shape, aa_fore)
#     print('')
#     # print (top_)
#     # print('')
#     print(ix_.shape, ix_)
#     print('')
#     print(mesh_.shape, mesh_)
#     print('')
#     print(ixs_.shape, ixs_)
#
#     print ('')
#     print('')
#     print (a_del.shape, a_del)
#     print('')
#     print(b_del_.shape, b_del_)
#     print('')
#     print('ixs0 ', ixs_.shape, ixs_[0])
#     print ('')
#     print('fore ', fore_.shape, fore_)
#     print('')
#     print('fore_nw_ ', fore_nw_.shape, fore_nw_)
#     print('')




#
# print ('')
# print('')
# indices = tf.constant([[2,0,3], [2,0,3]])
# print (indices.shape[1])
# print(indices.shape[0])
# rng_1 = tf.range(3)
# rng_2 = tf.range(2)
# mesh = tf.meshgrid(rng_1, rng_2)[1]
#
# #Stack mesh and the idx
# full_indices = tf.stack([mesh, indices], axis=2) #tf.constant([1,2])#
#
# with tf.Session() as sess:
#     rng_1_, rng_2_, indices_, mesh_, full_indices_ = sess.run([rng_1, rng_2, indices, mesh, full_indices])
#     # print(a)
#     print('')
#     print ('rng_1_ ', rng_1_)
#     print ('')
#     print ('rng_2_ ',rng_2_)
#     print ('')
#     print('indices_ ',indices_)
#     print('')
#     print('mesh_ ', mesh_)
#     print('')
#     print('full_indices', full_indices_)
#
