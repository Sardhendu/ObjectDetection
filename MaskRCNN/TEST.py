
import tensorflow as tf
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

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


class ProposalLayer():
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """
    
    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        
        print('RUNNING ProposalLayer ......................')
    
    def call(self, inputs):
        logging.info('IN THE CALL FUNCTION OF ProposalLayer')
        logging.info('Length of inputs: %s', len(inputs))
        logging.info('Shape rpn_class: %s', inputs[0].shape)#get_shape().as_list())
        logging.info('Shape rpn_bbox: %s', inputs[1].shape)#get_shape().as_list())
        logging.info('Shape anchors: %s', inputs[2].shape)#get_shape().as_list())
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        logging.info('Shape scores : %s', scores.shape)#get_shape().as_list())
        
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STDDEV, [1, 1, 4])
        logging.info('Shape deltas (input[1]): %s', deltas.shape)#get_shape().as_list())
        # Anchors
        anchors = inputs[2]
        
        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        logging.info('tf.shape(anchors)[1] = %s', tf.shape(anchors)[1])
        pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[1])
        logging.info('Shape pre_nms_limit : %s', pre_nms_limit.get_shape().as_list())
        
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        logging.info('Shape ix : %s', ix.get_shape().as_list())
        
        scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y), 1)
        logging.info('Shape scores after batch_slice: %s', scores.get_shape().as_list())
        
        deltas = batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), 1)
        logging.info('Shape deltas after batch_slice: %s', deltas.get_shape().as_list())
        
        pre_nms_anchors = batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), 1,
                                            names=["pre_nms_anchors"])
        logging.info('Shape pre_nms_anchors after batch_slice: %s', pre_nms_anchors.get_shape().as_list())
        
        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y), 1,
                                  names=["refined_anchors"])
        logging.info('Shape boxes after batch_slice: %s', boxes.get_shape().as_list())
        
        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window), 1,
                                  names=["refined_anchors_clipped"])
        logging.info('Clipped Boxes after batch_slice: %s', boxes.get_shape().as_list())
        
        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.
        
        # Non-max suppression
        def nms(boxes, scores):
            logging.info('IN THE CALL nms Function inside call function OF ProposalLayer')
            indices = tf.image.non_max_suppression(
                    boxes, scores, self.proposal_count,
                    self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = batch_slice([boxes, scores], nms, 1)
        
        logging.info('Proposal shape after batch_slice: %s', proposals.get_shape().as_list())
        
        return proposals
    
    def compute_output_shape(self, input_shape):
        logging.info('IN THE compute_output_shape function OF ProposalLayer')
        return (None, self.proposal_count, 4)



from MaskRCNN.config import config
np.random.seed(325)
num_batches = 2
proposal_count = 2
nms_threshold = 0.3

rpn_class_probs = np.array(np.random.random((3,5,2)), dtype='float32')
rpn_bbox = np.array(np.random.random((3, 5, 4)), dtype='float32')
input_anchors = np.array(np.random.random((3, 5, 4)), dtype='float32')

proposals = ProposalLayer(proposal_count, nms_threshold, config).call([rpn_class_probs, rpn_bbox, input_anchors])