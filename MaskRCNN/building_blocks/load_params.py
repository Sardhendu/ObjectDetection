
import logging
import numpy as np
import tensorflow as tf
import h5py

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def _convert_string_dtype(dtype):
    if dtype == 'float16':
        return tf.float16
    if dtype == 'float32':
        return tf.float32
    elif dtype == 'float64':
        return tf.float64
    elif dtype == 'int16':
        return tf.int16
    elif dtype == 'int32':
        return tf.int32
    elif dtype == 'int64':
        return tf.int64
    elif dtype == 'uint8':
        return tf.int8
    elif dtype == 'uint16':
        return tf.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)

def set_value(sess, tensor_variable, value):
    
    """
    Refer: https://stackoverflow.com/questions/43016565/tensorflow-re-initializing-weights-and-reshaping-tensor-of
    -pretrained-model
    Sets the value of a variable, from a Numpy array.
    # Arguments
        x: Tensor to set to a new value.
        value: Value to set the tensor to, as a Numpy array
            (of the same shape).
    """
    value = np.asarray(value)
    tf_dtype = _convert_string_dtype(tensor_variable.dtype.name.split('_')[0])
    if hasattr(tensor_variable, '_assign_placeholder'):
        assign_placeholder = tensor_variable._assign_placeholder
        assign_op = tensor_variable._assign_op
    else:
        assign_placeholder = tf.placeholder(tf_dtype, shape=value.shape)
        assign_op = tensor_variable.assign(assign_placeholder)
        tensor_variable._assign_placeholder = assign_placeholder
        tensor_variable._assign_op = assign_op
    return sess.run(assign_op, feed_dict={assign_placeholder: value})


def set_pretrained_weights(sess, weights_path):
    pretrained_weights = h5py.File(weights_path, mode='r')

    if h5py is None:
        raise ImportError('load_weights requires h5py.')
    
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        scope_name, graph_var_name = var.name.split('/')

        try:
            if scope_name.split('_')[0] == 'rpn':
                pretrained_var_name = pretrained_weights['rpn_model'][scope_name]
            else:
                pretrained_var_name = pretrained_weights[scope_name][scope_name]

            if graph_var_name == 'kernel:0':
                val = pretrained_var_name['kernel:0']
                logging.info('w:0, scope_name = %s, pre-trained.shape = %s, variable.shape = %s', str(scope_name), str(val.value.shape), str(var.shape))

            elif graph_var_name == 'bias:0':
                val = pretrained_var_name['bias:0']
                logging.info('b:0, scope_name = %s, pre-trained.shape = %s, variable.shape = %s', str(scope_name), str(val.value.shape), str(var.shape))

            elif graph_var_name == 'moving_mean:0':
                val = pretrained_var_name['moving_mean:0']
                logging.info('m:0, scope_name = %s, pre-trained.shape = %s, variable.shape = %s', str(scope_name), str(val.value.shape), str(var.shape))

            elif graph_var_name == 'moving_variance:0':
                val = pretrained_var_name['moving_variance:0']
                logging.info('v:0, scope_name = %s, pre-trained.shape = %s, variable.shape = %s', str(scope_name), str(val.value.shape), str(var.shape))

            elif graph_var_name == 'beta:0':
                val = pretrained_var_name['beta:0']
                logging.info('beta:0, scope_name = %s, pre-trained.shape = %s, variable.shape = %s', str(scope_name), str(val.value.shape), str(var.shape))

            elif graph_var_name == 'gamma:0':
                val = pretrained_var_name['gamma:0']
                logging.info('gamma:0, scope_name = %s, pre-trained.shape = %s, variable.shape = %s', str(scope_name), str(val.value.shape), str(var.shape))

            # elif graph_var
            else:
                logging.info ('############### %s', graph_var_name)
                val = None
                var = None

            if val.value.shape != var.shape:
                raise ValueError('Mismatch is shape of pretrained weights and network defined weights')
            #
            # if graph_var_name == 'b:0':
            #     print (val.value)
            #     print ('.........................')
            #     print (sess.run(var))
            #     print ('loading pretrained weights for variable')
            set_value(sess=sess, tensor_variable=var, value=val)
                    # print ('Loaded')
                    # print(sess.run(var))

            # for keyy, vv in pretrained_weights[scope_name].items():
            #     print (keyy, vv)
        except KeyError:
            print('OOPS variable %s not found in pretrained variable list '%(str(scope_name)))
            
    pretrained_weights.close()


def debugg():
    import h5py
    weight_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
    pretrained_weights = h5py.File(weight_path, 'r')
    print (pretrained_weights.name)

    # print (pretrained_weights['mrcnn_class_conv1']['mrcnn_class_conv1']['kernel:0'].value.shape)
    # print(pretrained_weights['mrcnn_class_conv2']['mrcnn_class_conv2']['kernel:0'].value.shape)
    # for k, v in pretrained_weights.items():
    #     # if k == 'mrcnn_mask_conv1':
    #     print (k, v.keys())
    #     print(v.get('/mrcnn_mask_conv1'))
    #     for k1, v1 in v.items():
    #         print (k1)
    #         print (v1)
    #         for k2, v2 in v1.items():
    #             print(k2)
    #             print(v2.value.shape)
    # # print(pretrained_weights['mrcnn_class_conv1']['mrcnn_class_conv1']['mrcnn_class_conv1'])
    # # print("Keys: %s" % pretrained_weights.keys())
    # # print (pretrained_weights['ROI']['mrcnn_class_conv1'])
    # for k, v in pretrained_weights.items():
    #     # print (k, v)
    #     if k == 'mrcnn_class_conv1':
    #         # print (v.shape)
    #         # print (k)
    #         # print ('')
    #         for k1, v1 in v.items():
    #             print ('sdfsdfsd', k1, v1)
    #         print('adadadas')
        
    # a_group_key = list(f.keys())[0]
    # print (a_group_key)
    # with tf.Session() as sess:
    #     weight_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
    #     sess.run(tf.global_variables_initializer())
    #     set_pretrained_weights(sess, weights_path=weight_path)
        
debugg()