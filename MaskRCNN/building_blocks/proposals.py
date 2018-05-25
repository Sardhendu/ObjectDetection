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
    logging.info('anchors shape: %s', str(tf.shape(anchors)))
    
    # Searching through lots of anchors can be time consuming. So we would select at most 6000 of them for further processing
    max_anc_before_nms = tf.minimum(2, tf.shape(anchors)[1])
    
    # Here we fetch the idx of the top 6000 anchors
    ix = tf.nn.top_k(foreground_probs, max_anc_before_nms, sorted=True, name="top_anchors").indices
    
    # Now that we have the idx of the top anchors we would want to only gather the data related pertaining to those idx. We would wanna gather foreground_prob and box_delta only for the selected anchors.

    foreground_probs2 = tf.gather_nd(foreground_probs, ix)
    
    
    return foreground_probs, box_delta, anchors, max_anc_before_nms, ix, foreground_probs2
    
    
def debugg():
    from MaskRCNN.config import config as conf

    rpn_probs = tf.placeholder(dtype=tf.float32,
                                   shape=[1, 3, 2],
                                   name="rpn_prob")

    rpn_box = tf.placeholder(dtype=tf.float32,
                                   shape=[1, 3, 4],
                                   name="rpn_box")


    input_anchors = tf.placeholder(dtype=tf.float32,
                                   shape=[1, 4, 4],
                                   name="input_anchors")
    foreground_probs, box_delta, anchors, max_anc_before_nms, ix , foreground_probs2 = proposals(conf, inputs=[rpn_probs, rpn_box, input_anchors])
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        a= np.random.random((1, 3, 2))
        b= np.random.random((1, 3, 4))
        c= np.random.random((1, 4, 4))
        feed_dict={rpn_probs: a, rpn_box: b, input_anchors: c}

        print('rpn_probs ', a)
        print('')
        print('rpn_box ', b)
        print('')
        print('input_anchors ', c)
        fp, bxd, anch, anch_bft_nms, ix_, fp2 = sess.run([foreground_probs, box_delta, anchors, max_anc_before_nms, ix, foreground_probs2 ], feed_dict=feed_dict)
        
        print ('')
        print('')
        print('foreground_probs = ', foreground_probs.shape, fp)
        print('')
        print('box_delta = ', bxd.shape, bxd)
        print('')
        print('anchors = ', anch.shape, anch)
        print('')
        print('max_anc_before_nms = ', anch_bft_nms.shape, anch_bft_nms)
        print('')
        print('ix_ = ', ix_)

        # print('')
        # print('fp2 = ', fp2)


# debugg()
#
def get_slices_given_idx(inputs, batch_size):
    for i in range(0,batch_size):
        inputs_slice = [x[i] for x in inputs]


np.random.seed(675)
a = tf.placeholder(dtype=tf.float32, shape=(2,4,2), name='a')
a_fore = a[:,:,1]

a_delta = tf.placeholder(dtype=tf.float32, shape=(2,4,4), name='a_del')
top = tf.nn.top_k(a_fore, 2, sorted=True, name="top_anchors")
ix = tf.nn.top_k(a_fore, 2, sorted=True, name="top_anchors").indices
# po = [[a_fore, ix], lambda x, y: tf.gather_nd(x, y)]

# For Foreground_prob
hmm = [[[0,2],[0,0]], [[1,2],[1,0]]]
b = tf.gather_nd(a_fore, hmm)
b_del = tf.gather_nd(a_delta, hmm)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    aa, a_del, aa_fore, top_, ix_, bb, bb_del= \
        sess.run([a,a_delta, a_fore,top, ix, b,b_del],
                 feed_dict={a:np.random.random((2,4,2)),
                            a_delta:np.random.random((2,4,4))})
    print(aa.shape, aa)
    print('')
    print(aa_fore.shape, aa_fore)
    print('')
    print (top_)
    print('')
    print(ix_.shape, ix_)
    print('')
    print(bb.shape, bb)

    print ('')
    print('')
    print (a_del.shape, a_del)
    print('')
    print(bb_del.shape, bb_del)

    # print ('')
    # print ('ppo ',ppo)
    
    # for i in ix

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