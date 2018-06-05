'''

1. We have features map in different shapes ranging from 32 to 256 dim
2. This bring the problem of different scales, regardless of the scale we should be assign the anchor box to the right location in the image. Here we take care of that.
'''


import tensorflow as tf
import numpy as np

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)

def assign_boxes_debugg(boxes, image_shape):
    # Image meta
    # Holds details about the image. See compose_image_meta()
    
    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    # feature_maps = inputs[2:]
    
    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
    h = y2 - y1
    w = x2 - x1
    # Use shape of first image. Images in a batch must have the same size.
    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
    aa = tf.sqrt(h * w)
    bb = tf.sqrt(image_area)
    roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
    roi_level = 4 + tf.cast(tf.round(roi_level), tf.int32)
    roi_level = tf.minimum(5, tf.maximum(
            2, roi_level))
    roi_level = tf.squeeze(roi_level, 2)
    
    return aa, bb, roi_level
    
    # # Loop through levels and apply ROI pooling to each. P2 to P5.
    # pooled = []
    # box_to_level = []
    # for i, level in enumerate(range(2, 6)):
    #     ix = tf.where(tf.equal(roi_level, level))
    #     level_boxes = tf.gather_nd(boxes, ix)
    #
    #     # Box indicies for crop_and_resize.
    #     box_indices = tf.cast(ix[:, 0], tf.int32)
    #
    #     # Keep track of which box is mapped to which level
    #     box_to_level.append(ix)
    #
    #     # Stop gradient propogation to ROI proposals
    #     level_boxes = tf.stop_gradient(level_boxes)
    #     box_indices = tf.stop_gradient(box_indices)
    #
    #     # Crop and Resize
    #     # From Mask R-CNN paper: "We sample four regular locations, so
    #     # that we can evaluate either max or average pooling. In fact,
    #     # interpolating only a single value at each bin center (without
    #     # pooling) is nearly as effective."
    #     #
    #     # Here we use the simplified approach of a single value per bin,
    #     # which is how it's done in tf.crop_and_resize()
    #     # Result: [batch * num_boxes, pool_height, pool_width, channels]
    #     pooled.append(tf.image.crop_and_resize(
    #             feature_maps[i], level_boxes, box_indices, pool_shape,
    #             method="bilinear"))
    #
    # # Pack pooled features into one tensor
    # pooled = tf.concat(pooled, axis=0)
    #
    # # Pack box_to_level mapping into one array and add another
    # # column representing the order of pooled boxes
    # box_to_level = tf.concat(box_to_level, axis=0)
    # box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
    # box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
    #                          axis=1)
    #
    # # Rearrange pooled features to match the order of the original boxes
    # # Sort box_to_level by batch then box index
    # # TF doesn't have a way to sort by two columns, so merge them and sort.
    # sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
    # ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
    #         box_to_level)[0]).indices[::-1]
    # ix = tf.gather(box_to_level[:, 2], ix)
    # pooled = tf.gather(pooled, ix)
    #
    # # Re-add the batch dimension
    # pooled = tf.expand_dims(pooled, 0)
    # return pooled

def assign_boxes(boxes, layers, image_shape):
    '''
    
    :param boxes: Normalized anchor boxes (Proposals)
    :param layers: [2, 3, 4, 5] , pertaining to P2, P3, P4, P5
    :return:
    '''
    k0 = 4
    min_k = min(layers)
    max_k = max(layers)
    
    h = boxes[:,:,2] - boxes[:,:,0]
    w = boxes[:,:,3] - boxes[:,:,1]
    # print(h)
    # print(w)
    areas = h*w
    # areas = areas.reshape([areas.shape[0], areas.shape[1], 1]) # [num_batches, num_boxes, 1]
    image_area = image_shape[0] * image_shape[1]
    print (areas)

    # We use the formula given in the FPN (Feature Pyramid Network paper), this formula has another interesting term
    # which is the multiplication of np.sqrt(image_area), this is done because our boxes are still in normalized
    # coordinates
    print (np.sqrt(areas))
    print (np.sqrt(image_area))
    k = k0 + np.round(np.log2(np.sqrt(areas) / 224 * np.sqrt(image_area)))  # The paper says to perform
    # floor operation

    # k = np.minimum(5, np.maximum(2, k))

    print (k)
    #
    #


    
    
def debugg():
    from MaskRCNN.config import config as conf
    from MaskRCNN.building_blocks.proposals_tf import Proposals

    np.random.seed(325)
    num_batches = 3

    a = np.array(np.random.random((3, 5, 2)), dtype='float32')
    b = np.array(np.random.random((3, 5, 4)), dtype='float32')
    c = np.array(np.random.random((3, 5, 4)), dtype='float32')

    obj_p = Proposals(conf, inference_batch_size=num_batches)
    p_graph = obj_p.get_proposal_graph()
    ancd = obj_p.get_anchors_delta()
    
    feed_dict = {p_graph['rpn_probs']: a, p_graph['rpn_bbox']: b, p_graph['input_anchors']: c}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        p_ = sess.run(p_graph['proposals'], feed_dict=feed_dict)

        print('')
        print(p_.shape)
        print('')
        print (p_)
        print ('')
        assign_boxes(p_, [2,3,4,5], image_shape=[800, 1024])

        roi_level = assign_boxes_debugg(boxes=p_, image_shape=[800, 1024])
        
        aa_, bb_, r_l = sess.run(roi_level)
        print ('')
        print('TENSORFLOW STUFF.........')
        print (aa_)
        print (bb_)
        print (r_l)

debugg()








'''

[[-0.61580211  0.59458905]
 [ 0.03991634 -0.36171046]
 [-0.03470951 -0.58450645]]
[[-0.35293397  0.84925073]
 [-0.53971034 -0.53223413]
 [-0.09493041 -0.25965157]]
[[[ 0.21733749]
  [ 0.50495517]]

 [[ 0.02154326]
  [ 0.19251466]]

 [[ 0.00329499]
  [ 0.15176801]]]
[ 4.  5.  3.  4.  1.  4.]
[ 4.  5.  3.  4.  2.  4.]'''