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
    
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(
                dtype=tf.float32,
                shape=k_shape,
                initializer=wght_init,
                name="w",
                trainable=True
        )
        bias = tf.get_variable(
                dtype=tf.float32,
                shape=[k_shape[-1]],
                initializer=tf.constant_initializer(1),
                name='b',
                trainable=True
        )
    
    if w_decay:
        weight_decay = tf.multiply(tf.nn.l2_loss(weight), w_decay, name='weight_loss')
        tf.add_to_collection('loss_w_decay', weight_decay)
    
    
    if add_smry:
        tf.summary.histogram("conv_weights", weight)
        tf.summary.histogram("conv_bias", bias)
    
    return tf.nn.conv2d(X, weight, [1, stride, stride, 1], padding=padding) + bias


def conv2D_transposed_strided(X, k_shape, stride=2, padding='SAME', w_init='tn', out_shape=None, scope_name='conv_layer', add_smry=False):
    '''
    :param X:           The input
    :param k_shape:     The shape for weight filter
    :param stride:      Strides, It should take the value 2 if upsampling double
    :param padding:     Should be same
    :param w_init:      which weight initializer (Glorot, truncated etc.)
    :param out_shape:   The output shape of the upsampled data
    :param scope_name:
    :param add_smry:
    :return:
    '''

    if w_init == 'gu':
        wght_init = tf.glorot_uniform_initializer()
    else:
        wght_init = tf.truncated_normal_initializer(
                stddev=0.1
        )
    
    hght, wdth, in_ch, out_ch = k_shape
    
    with tf.variable_scope(scope_name):
        weight = tf.get_variable(
                dtype=tf.float32,
                shape=[hght, wdth, out_ch, in_ch],  # Note : We swap the in_out_channels
                initializer=wght_init,
                name="w",
                trainable=True
        )
        bias = tf.get_variable(
                dtype=tf.float32,
                shape=out_ch,  # k_shape[-1],
                initializer=tf.constant_initializer(1),
                name='b',
                trainable=True
        )
    
    
    if add_smry:
        tf.summary.histogram("dconv_weights", weight)
        tf.summary.histogram("dconv_bias", bias)
    # print ('dasdasdas', X.shape)
    # print (weight.shape, bias.shape)
    if out_shape is None:
        out_shape = list(X.get_shape().as_list())
        out_shape[1] *= 2  # Should be doubled when upsampling
        out_shape[2] *= 2
        out_shape[3] = out_ch  # weight.get_shape().as_list()[3]
    # print (out_shape)
    conv = tf.nn.conv2d_transpose(X, weight,
                                  tf.stack([tf.shape(X)[0], int(out_shape[1]), int(out_shape[2]), int(out_shape[3])]),
                                  strides=[1, stride, stride, 1],
                                  padding=padding)
    # print (conv.shape)
    return tf.nn.bias_add(conv, bias)


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
                name="m",  # offset (bias)
                trainable=False)
        
        expBatchVar_avg = tf.get_variable(
                dtype='float32',
                shape=[numOUT],
                initializer=tf.constant_initializer(1.0),
                name="v",  # scale(weight)
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
                                 name='w',
                                 trainable=True
                                 )
        bias = tf.get_variable(dtype=tf.float32,
                               shape=[k_shape[-1]],
                               initializer=tf.constant_initializer(1.0),
                               name='b',
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
    else:
        raise ValueError('Provide proper Activation function')