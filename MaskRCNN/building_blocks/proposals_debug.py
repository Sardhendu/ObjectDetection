
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
        self.batch_size = 3
        self.num_box_before_nms = 5
        self.rpn_class_probs = tf.placeholder(dtype=tf.float32,
                                              shape=[None, None, 2],
                                              name="rpn_prob")

        self.rpn_bbox = tf.placeholder(dtype=tf.float32,
                                       shape=[None, None, 4],
                                       name="rpn_bbox")

        self.input_anchors = tf.placeholder(dtype=tf.float32,
                                            shape=[None, None, 4],
                                            name="input_anchors")
        
        print('RUNNING ProposalLayer ......................')
    
    def call(self):
        logging.info('IN THE CALL FUNCTION OF ProposalLayer')
        logging.info('Shape rpn_class: %s', self.rpn_class_probs.get_shape().as_list())
        logging.info('Shape rpn_bbox: %s', self.rpn_bbox.get_shape().as_list())
        logging.info('Shape anchors: %s', self.input_anchors.get_shape().as_list())
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = self.rpn_class_probs[:, :, 1]
        logging.info('Shape scores : %s', scores.get_shape().as_list())
        
        # Box deltas [batch, num_rois, 4]
        deltas = self.rpn_bbox
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STDDEV, [1, 1, 4])
        logging.info('Shape deltas (input[1]): %s', deltas.get_shape().as_list())
        # Anchors
        anchors = self.input_anchors
        
        print (scores.dtype, deltas.dtype, anchors.dtype)
        
        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        logging.info('tf.shape(anchors)[1] = %s', tf.shape(anchors)[1])
        pre_nms_limit = tf.minimum(self.num_box_before_nms, tf.shape(anchors)[1])
        logging.info('Shape pre_nms_limit : %s', pre_nms_limit.get_shape().as_list())
        
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        logging.info('Shape ix : %s', ix.get_shape().as_list())
        
        scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.batch_size)
        logging.info('Shape scores after batch_slice: %s', scores.get_shape().as_list())
        
        deltas = batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.batch_size)
        logging.info('Shape deltas after batch_slice: %s', deltas.get_shape().as_list())

        anchors = batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.batch_size,
                                            names=["pre_nms_anchors"])
        logging.info('Shape anchors after batch_slice: %s', anchors.get_shape().as_list())
        
        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        anchor_delta = batch_slice([anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y), self.batch_size,
                                  names=["refined_anchors"])
        logging.info('Shape boxes after batch_slice: %s', anchor_delta.get_shape().as_list())
        
        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        anchor_delta_clipped = batch_slice(anchor_delta,
                                  lambda x: clip_boxes_graph(x, window), self.batch_size,
                                  names=["refined_anchors_clipped"])
        logging.info('Clipped Boxes after batch_slice: %s', anchor_delta_clipped.get_shape().as_list())
        
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
        proposals = batch_slice([anchor_delta_clipped, scores], nms, self.batch_size)
        
        logging.info('Proposal shape after batch_slice: %s', proposals.get_shape().as_list())
        
        return (self.rpn_class_probs, self.rpn_bbox, self.input_anchors, proposals,
                dict(ix=ix, scores=scores, bbox_delta_=deltas, anchors=anchors, anchor_delta=anchor_delta, anchor_delta_clipped=anchor_delta_clipped))
    
    def compute_output_shape(self, input_shape):
        logging.info('IN THE compute_output_shape function OF ProposalLayer')
        return (None, self.proposal_count, 4)
    
    



from MaskRCNN.config import config
np.random.seed(325)
num_batches = 3
proposal_count = 4
nms_threshold = np.float32(0.3)

a = np.array(np.random.random((3, 5, 2)), dtype='float32')
b = np.array(np.random.random((3, 5, 4)), dtype='float32')
c = np.array(np.random.random((3, 5, 4)), dtype='float32')

rpn_class_probs, rpn_bbox, input_anchors, proposals, others = \
    ProposalLayer(proposal_count, nms_threshold, config).call()
# print(proposals)
feed_dict={rpn_class_probs: a, rpn_bbox: b, input_anchors: c}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    bbox_delta_ = sess.run(others['bbox_delta_'], feed_dict=feed_dict)
    ix_ = sess.run(others['ix'], feed_dict = feed_dict)
    scores_ = sess.run(others['scores'], feed_dict=feed_dict)
    anchors_ = sess.run(others['anchors'], feed_dict=feed_dict)
    anchor_delta_ = sess.run(others['anchor_delta'], feed_dict=feed_dict)
    anchor_delta_clipped_ = sess.run(others['anchor_delta_clipped'], feed_dict=feed_dict)
    proposals_ = sess.run(proposals, feed_dict=feed_dict)

    print('bbox_delta_ ', bbox_delta_.shape, bbox_delta_)
    print('')
    print('ix_', ix_.shape, ix_)
    print('')
    print('scores_ ', scores_.shape, scores_)
    print('')
    print('anchors_ ', anchors_.shape, anchors_)
    print('')
    print('anchor_delta_ ', anchor_delta_.shape, anchor_delta_)
    print('')
    print('anchor_delta_clipped_ ', anchor_delta_clipped_.shape, anchor_delta_clipped_)
    print('')
    print('proposals_ ', proposals_.shape, proposals_)
    print('')



'''
bbox_delta_  (3, 5, 4) [[[  6.08330965e-02   6.71194121e-02   9.41824615e-02   1.71722159e-01]
  [  3.85359898e-02   9.32771638e-02   6.52525723e-02   1.92653507e-01]
  [  6.77995607e-02   6.24737255e-02   1.89149845e-02   1.74794272e-02]
  [  8.25559869e-02   4.15760539e-02   1.90824702e-01   7.37159774e-02]
  [  2.36403178e-02   4.84634563e-02   1.80499837e-01   8.22036862e-02]]

 [[  1.43593876e-02   1.00482712e-02   5.03815077e-02   3.92356282e-03]
  [  7.38601610e-02   2.49234634e-03   1.27579302e-01   6.00859821e-02]
  [  5.37443422e-02   3.63819599e-02   1.18435375e-01   7.35748857e-02]
  [  5.09575866e-02   8.80408734e-02   1.45163164e-01   1.63295969e-01]
  [  5.10929599e-02   9.09873471e-02   9.35459062e-02   1.09360747e-01]]

 [[  9.55238715e-02   1.13850981e-02   1.28908351e-01   2.66324226e-02]
  [  7.86226615e-02   9.94997323e-02   1.70552894e-01   7.04915524e-02]
  [  9.70137939e-02   7.02699199e-02   1.12953506e-01   4.68629441e-05]
  [  1.20924581e-02   2.32591256e-02   1.77218363e-01   1.01405002e-01]
  [  3.07332613e-02   3.16062346e-02   1.03429772e-01   1.94887802e-01]]]

ix_ (3, 5) [[0 3 2 1 4]
 [1 2 0 4 3]
 [4 0 2 3 1]]

scores_  (3, 5) [[ 0.98104852  0.79038447  0.76019788  0.68306577  0.48757792]
 [ 0.95836937  0.92400974  0.55117744  0.54641497  0.37665966]
 [ 0.92985797  0.9152084   0.90162092  0.88823181  0.60490471]]

anchors_  (3, 5, 4) [[[ 0.66516078  0.7107172   0.104709    0.41347158]
  [ 0.4026624   0.00647369  0.97270262  0.70907563]
  [ 0.36219054  0.18682894  0.75377899  0.75660789]
  [ 0.0971365   0.30265555  0.30198509  0.8906796 ]
  [ 0.49675083  0.36046994  0.825221    0.13738894]]

 [[ 0.51335049  0.67271036  0.55130559  0.13511348]
  [ 0.45991206  0.69495296  0.141526    0.19375683]
  [ 0.71754569  0.79186046  0.89333463  0.11504316]
  [ 0.17198341  0.0888403   0.83701622  0.88303244]
  [ 0.27573726  0.1545323   0.95328003  0.57610214]]

 [[ 0.40544087  0.96145767  0.37492931  0.86902213]
  [ 0.63724017  0.69959635  0.14438523  0.45761779]
  [ 0.50528699  0.13827209  0.50424409  0.81720257]
  [ 0.2975572   0.38713291  0.40366048  0.97275996]
  [ 0.5266065   0.22203949  0.45695686  0.74100089]]]

anchor_delta_  (3, 5, 4) [[[ 0.65874195  0.71861047  0.04293984  0.36567649]
  [ 0.40541095 -0.00262791  1.01388812  0.84925073]
  [ 0.38500136  0.21740168  0.78406721  0.79722756]
  [ 0.09251355  0.30461103  0.34043097  0.93761951]
  [ 0.47202766  0.35921511  0.86547446  0.11702123]]

 [[ 0.5129149   0.66836518  0.55283123  0.12865484]
  [ 0.45805818  0.70922279  0.09634772  0.17698866]
  [ 0.71594203  0.79307377  0.91383362  0.06458205]
  [ 0.15392342  0.08832273  0.92285311  1.02339268]
  [ 0.27713463  0.16853054  1.02111793  0.63881898]]

 [[ 0.40462527  0.96165276  0.36991575  0.86672235]
  [ 0.64431632  0.68435603  0.05980986  0.42470446]
  [ 0.50524819  0.18596455  0.50408059  0.86492682]
  [ 0.28855395  0.36950359  0.41522977  1.01763153]
  [ 0.52826071  0.18260825  0.45102149  0.81323701]]]

anchor_delta_clipped_  (3, 5, 4) [[[ 0.65874195  0.71861047  0.04293984  0.36567649]
  [ 0.40541095  0.          1.          0.84925073]
  [ 0.38500136  0.21740168  0.78406721  0.79722756]
  [ 0.09251355  0.30461103  0.34043097  0.93761951]
  [ 0.47202766  0.35921511  0.86547446  0.11702123]]

 [[ 0.5129149   0.66836518  0.55283123  0.12865484]
  [ 0.45805818  0.70922279  0.09634772  0.17698866]
  [ 0.71594203  0.79307377  0.91383362  0.06458205]
  [ 0.15392342  0.08832273  0.92285311  1.        ]
  [ 0.27713463  0.16853054  1.          0.63881898]]

 [[ 0.40462527  0.96165276  0.36991575  0.86672235]
  [ 0.64431632  0.68435603  0.05980986  0.42470446]
  [ 0.50524819  0.18596455  0.50408059  0.86492682]
  [ 0.28855395  0.36950359  0.41522977  1.        ]
  [ 0.52826071  0.18260825  0.45102149  0.81323701]]]

proposals_  (3, 4, 4) [[[ 0.65874195  0.71861047  0.04293984  0.36567649]
  [ 0.40541095  0.          1.          0.84925073]
  [ 0.47202766  0.35921511  0.86547446  0.11702123]
  [ 0.          0.          0.          0.        ]]

 [[ 0.5129149   0.66836518  0.55283123  0.12865484]
  [ 0.45805818  0.70922279  0.09634772  0.17698866]
  [ 0.71594203  0.79307377  0.91383362  0.06458205]
  [ 0.15392342  0.08832273  0.92285311  1.        ]]

 [[ 0.40462527  0.96165276  0.36991575  0.86672235]
  [ 0.64431632  0.68435603  0.05980986  0.42470446]
  [ 0.50524819  0.18596455  0.50408059  0.86492682]
  [ 0.28855395  0.36950359  0.41522977  1.        ]]]

'''
