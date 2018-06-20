

import tensorflow as tf
import logging


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    print('RUNNING utils (batch_slice)......................')
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    logging.info('outputs shape,..... %s', str(len(outputs)))
    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]
    logging.info('result shape,..... %s', str(result.shape))
    return result


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """

    print('RUNNING apply_box_deltas_graph ......................')
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """

    print('RUNNING clip_boxes_graph ......................')
    # Split
    logging.info ('Inside: clip_boxes_graph boxes.shape = %s', str(boxes.shape))
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    print('RUNNING refine_detections_graph ......................')
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_THRESHOLD:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_THRESHOLD)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_POST_NMS_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_POST_NMS_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_POST_NMS_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_POST_NMS_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_POST_NMS_INSTANCES
    gap = config.DETECTION_POST_NMS_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

class DetectionLayer():
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """
    
    def __init__(self, config, image_shape, image_window):
        self.config = config
        self.image_shape = image_shape
        self.image_window = image_window
        print('RUNNING DetectionLayer ......................')
    
    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        
        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        # image_shape = self.image_shape[0]
        window = norm_boxes_graph(self.image_window, self.image_shape[:2])
        
        # Run detection refinement graph on each item in the batch
        detections_batch = batch_slice(
                [rois, mrcnn_class, mrcnn_bbox, window],
                lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),1)
        
        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_score)] in
        # normalized coordinates
        return tf.reshape(
                detections_batch,
                [1, self.config.DETECTION_POST_NMS_INSTANCES, 6])





def debug():
    import numpy as np
    from MaskRCNN.config import config as conf
    np.random.seed(982)
    proposals = np.array(np.random.random((1, 8, 4)), dtype='float32')
    mrcnn_class_probs = np.array(np.random.random((1, 8, 4)), dtype='float32')  # [num_batch, num_top_proposal,
    # num_classes]
    mrcnn_bbox = np.array(np.random.random((1, 8, 4, 4)), dtype='float32')
    window = np.array([[131, 0, 893, 1155]], dtype='int32')  # image without zeropad [y1, x1, y2,
    # #  x2]
    
    # window = np.array([131, 0, 893, 1155], dtype='int32')
    
    print('mrcnn_class_probs ', mrcnn_class_probs.shape, mrcnn_class_probs)
    print('')
    print('mrcnn_bbox ', mrcnn_bbox)
    print('')
    
    obj_D = DetectionLayer(conf, [1024, 1024, 3], window)
    detections = obj_D.call([proposals, mrcnn_class_probs, mrcnn_bbox])
    # (class_ids, indices, mesh, ixs, class_scores, bbox_delta, refined_proposals, class_id_idx, score_id_idx,
    #  keep_idx, pre_nms_class_ids, pre_nms_scores, pre_nms_porposals, unique_pre_nms_class_ids, class_nms_idx,
    #  post_nms_keep_idx, post_nms_scores, post_nms_topk_keep_idx, detection_per_batch) = obj_D.debug_outputs()
    #
    # detections = obj_D.get_detections()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        (detections_) =  sess.run(detections)  # ,
        # class_scores,
        #  bbox])
        # print('class_ids_ ', class_ids_.shape, class_ids_)
        # print('')
        # print('indices_ ', indices_)
        # print('')
        # print('mesh_ ', mesh_)
        # print('')
        # print('ixs_', ixs_.shape, ixs_)
        # print('')
        # print('class_scores_ ', class_scores_.shape, class_scores_)
        # print('')
        # print('bbox_delta_ ', bbox_delta_.shape, bbox_delta_)
        # print('')
        # print('refined_proposals_ ', refined_proposals_.shape, refined_proposals_)
        # print('')
        # print('class_id_idx_ ', class_id_idx_)
        # print('')
        # print('score_id_idx_ ', score_id_idx_)
        # print('')
        # print('keep_idx_ ', keep_idx_)
        # print('')
        # print('pre_nms_class_ids_ ', pre_nms_class_ids_.shape, pre_nms_class_ids_)
        # print('')
        # print('pre_nms_scores_ ', pre_nms_scores_.shape, pre_nms_scores_)
        # print('')
        # print('pre_nms_porposals_ ', pre_nms_porposals_.shape, pre_nms_porposals_)
        # print('')
        # print('unique_pre_nms_class_ids_ ', unique_pre_nms_class_ids_.shape, unique_pre_nms_class_ids_)
        # print('')
        # print('class_nms_idx_ ', class_nms_idx_)
        # print('')
        # print('post_nms_keep_idx_ ', post_nms_keep_idx_.shape, post_nms_keep_idx_)
        # print('')
        # print('post_nms_scores_ ', post_nms_scores_.shape, post_nms_scores_)
        # print('')
        # print('post_nms_topk_keep_idx_ ', post_nms_topk_keep_idx_.shape, post_nms_topk_keep_idx_)
        # print('')
        # print('detection_per_batch_ ', detection_per_batch_.shape, detection_per_batch_)
        # print('')
        print('detections_ ', detections_.shape, detections_)


debug()