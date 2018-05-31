
import logging
import tensorflow as tf
from FasterRCNN.config import config as conf

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

class rpn():
    
    def __init__(self, mode, feature_map):
        self.mode = mode
        self.feature_map = feature_map
        
        self.RPN_KERNEL_SIZE = conf.RPN_KERNEL_SIZE
        self.RPN_OUT_CHANNEL = conf.RPN_OUT_CHANNEL
        self.ANCHOR_PER_LOCATION = conf.ANCHOR_PER_LOCATION
        
        if mode == 'train':
           self.weights = dict(
                   weight_conv = tf.get_variable(shape=[self.RPN_KERNEL_SIZE , self.RPN_KERNEL_SIZE , 512,self.RPN_OUT_CHANNEL],
                                initializer=tf.truncated_normal_initializer(stddev=0.001),
                                dtype=tf.float32, name='rpn_conv_w'),
                   weight_cls = tf.get_variable(shape=[1 , 1 , self.RPN_OUT_CHANNEL, 2 * conf.ANCHOR_PER_LOCATION],
                                initializer=tf.truncated_normal_initializer(stddev=0.001),
                                dtype=tf.float32, name='rpn_cls_w'),
                   weight_reg = tf.get_variable(shape=[1, 1, self.RPN_OUT_CHANNEL, 4 * conf.ANCHOR_PER_LOCATION],
                               initializer=tf.truncated_normal_initializer(stddev=0.001),
                               dtype=tf.float32, name='rpn_reg_w')
           )
           
        self.build()
    
    def rpn_class_probs(self):
        '''
        Basically, we assume to have 9 anchors so,
            1. self.shared = [batch, h, w, 18]  18 = 9*2, 2-> object or not, 9->anchors per location
            2. Inorder to perform softmax we need to have the shape of [batch, h, w, anchors, 2]
                We rehsape tensors to achieve it.
            3. Once the softmax is performed we convert the probs back to original shape i.e = [batch, h, w, 18]
        :return:
        '''
        rpn_class_scores = tf.nn.conv2d(self.shared, self.weights['weight_cls'], strides=[1, 1, 1, 1],
                                             padding='SAME', name='rpn_conv')
        logging.info('rpn_class_logits: %s', rpn_class_scores.get_shape().as_list())
        
        # RESHAPE
        shape = tf.shape(rpn_class_scores)#.get_shape().as_list()
        rpn_class_scores = tf.reshape(rpn_class_scores, [shape[0], shape[1], shape[2], -1, 2])
        logging.info('rpn_class_scores: %s', rpn_class_scores.get_shape().as_list())
        
        # PERFORM SOFTMAX CLASSIFICATION
        self.rpn_class_probs = tf.nn.softmax(rpn_class_scores, name='rpn_class_probs')
        logging.info('rpn_class_probs: %s', self.rpn_class_probs.get_shape().as_list())
        
        # CONVERT BACK TO ORIGINAL SHAPE
        self.rpn_class_probs = tf.reshape(self.rpn_class_probs, (shape[0], shape[1], shape[2], -1))
        logging.info('rpn_class_probs: %s', self.rpn_class_probs.get_shape().as_list())
    
    
    def rpn_bbox_regression(self):
        '''
        input: self.shared = [batches, h, w, 4 * num_anchors]  -> 4{cy, cx, h, w}
        :return:
        '''
        self.rpn_bbox = tf.nn.conv2d(self.shared, self.weights['weight_reg'], strides=[1, 1, 1, 1],
                                     padding='SAME', name='rpn_conv')
    
    def build(self):
        
        '''
        The input to this is the feature map of the feature vector created using the VGG net.
        We perform the standard convolution operation and RELU operation.
        
        Basically we slide a kernel on the feature map, which will produce a new convoluted feature map. Every pixel
        position at the new feature map is considered to be a anchor.
        :return:
        '''
        with tf.variable_scope('rpn_shared_conv'):
            self.shared = tf.nn.conv2d(self.feature_map, self.weights['weight_conv'], strides=[1,1,1,1], padding='SAME', name='rpn_conv')
            self.shared = tf.nn.relu(self.shared)
        
        with tf.variable_scope('rpn_class_probs'):
            self.rpn_class_probs()
            
        with tf.variable_scope('rpn_bbox'):
            self.rpn_bbox_regression()
            
        if self.mode == 'train':
            pass # Here we do other stuffs
            
    def get_rpn_class_probs(self):
        return self.rpn_class_probs
    
    def get_rpn_bbox(self):
        return self.rpn_bbox
    
    

            
def debugg():
    rpn(mode='train', feature_map=tf.placeholder(shape=[None, 14, 14, 512], dtype=tf.float32))

# debugg()