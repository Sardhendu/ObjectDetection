import tensorflow as tf
from MaskRCNN.config import config as conf

def conv_layer(X, k_shape, stride=1, padding='SAME', w_init='tn', w_decay=None, scope_name='conv_layer',
               add_smry=False):
    
    if w_init == 'gu':
        wght_init = tf.glorot_uniform_initializer()
    else:
        wght_init = tf.truncated_normal_initializer(
                stddev=0.1
        )
    
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable(
                dtype=tf.float32,
                shape=k_shape,
                initializer=wght_init,
                name="kernel",
                trainable=True
        )
        bias = tf.get_variable(
                dtype=tf.float32,
                shape=[k_shape[-1]],
                initializer=tf.constant_initializer(1),
                name='bias',
                trainable=True
        )
    
    if w_decay:
        weight_decay = tf.multiply(tf.nn.l2_loss(weight), w_decay, name='weight_loss')
        tf.add_to_collection('loss_w_decay', weight_decay)
    
    
    if add_smry:
        tf.summary.histogram("conv_weights", weight)
        tf.summary.histogram("conv_bias", bias)
    
    return tf.nn.conv2d(X, weight, [1, stride, stride, 1], padding=padding) + bias


def batch_norm(X, axis=[0, 1, 2], scope_name=None):
    '''
    :param X:            The RELU output to be normalized
    :param numOUT:       Number of output channels (neurons)
    :param decay:        Exponential weighted average
    :param axis:         Normalization axis
    :param scope_name:
    :param trainable:
    :return:

    Why have: exponential decay with batch norm? In real sense taking the mean and variance of the complete training set
    makes more sense. Since we use batches we would wanna maintain a moving average of the mean and variance. This
    after many batches would approximately equal to the overall mean. Also during the test data collecting the mean
    and variance of the test data is not a good plan, what if test data is from  different distribution. So we apply
    the train mean and variance (calculated by moving average) to the test data too.
    '''
    decay = conf.BATCH_NORM_DECAY
    numOUT = X.get_shape().as_list()[-1]
    with tf.variable_scope(scope_name):
        beta = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(0.0),
                name="beta",  # offset (bias)
                trainable=True
        )
        gamma = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(1.0),
                name="gamma",  # scale(weight)
                trainable=True)
        
        expBatchMean_avg = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(0.0),
                name="moving_mean",  # offset (bias)
                trainable=False)
        
        expBatchVar_avg = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(1.0),
                name="moving_variance",  # scale(weight)
                trainable=False)
        
        batchMean, batchVar = tf.nn.moments(X, axes=axis, name="moments")
        trainMean = tf.assign(expBatchMean_avg,
                              decay * expBatchMean_avg + (1 - decay) * batchMean)
        trainVar = tf.assign(expBatchVar_avg,
                             decay * expBatchVar_avg + (1 - decay) * batchVar)
        
        with tf.control_dependencies([trainMean, trainVar]):
            bn = tf.nn.batch_normalization(X,
                                           mean=batchMean,
                                           variance=batchVar,
                                           offset=beta,
                                           scale=gamma,
                                           variance_epsilon=1e-5,
                                           name=scope_name)
        return bn


def fc_layers(X, k_shape, w_init='tn', scope_name='fc_layer', add_smry=False):

    
    if w_init == 'gu':
        wght_init = tf.glorot_uniform_initializer()
    else:
        wght_init = tf.truncated_normal_initializer(
                stddev=0.1
        )
    
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(dtype=tf.float32,
                                 shape=k_shape,
                                 initializer=wght_init,
                                 name='kernel',
                                 trainable=True
                                 )
        bias = tf.get_variable(dtype=tf.float32,
                               shape=[k_shape[-1]],
                               initializer=tf.constant_initializer(1.0),
                               name='bias',
                               trainable=True)
    
    
    X = tf.add(tf.matmul(X, weight), bias)
    
    if add_smry:
        tf.summary.histogram("fc_weights", weight)
        tf.summary.histogram("fc_bias", bias)
    return X


def activation(X, type='relu', scope_name='relu'):
    if type == 'relu':
        return tf.nn.relu(X, name=scope_name)
    elif type == 'sigmoid':
        return tf.nn.sigmoid(X, name=scope_name)
    elif type == 'tanh':
        return tf.nn.tanh(X, name=scope_name)
    elif type == 'softmax':
        return tf.nn.softmax(X, name=scope_name)
    else:
        raise ValueError('Provide proper Activation function')