import tensorflow as tf
import h5py


def check_params_consistency(weights_path):
    pretrained_weights = h5py.File(weights_path, mode='r')
    
    if h5py is None:
        raise ImportError('load_weights requires h5py.')
    
    for var in tf.get_collection(tf.GraphKeys.VARIABLES):
        print('............... ', var.name)
        scope_name, graph_var_name = var.name.split('/')
        try:
            if scope_name.split('_')[0] == 'rpn':
                pretrained_var_name = pretrained_weights['rpn_model'][scope_name]
            else:
                pretrained_var_name = pretrained_weights[scope_name][scope_name]

            if graph_var_name == 'w:0':
                val = pretrained_var_name['kernel:0']
                print('w:0', scope_name, val.value.shape, var.shape)

            elif graph_var_name == 'b:0':
                val = pretrained_var_name['bias:0']
                print('b:0', scope_name, val.value.shape, var.shape)

            elif graph_var_name == 'm:0':
                val = pretrained_var_name['moving_mean:0']
                print('m:0', scope_name, val.value.shape, var.shape)

            elif graph_var_name == 'v:0':
                val = pretrained_var_name['moving_variance:0']
                print('m:0', scope_name, val.value.shape, var.shape)

            elif graph_var_name == 'beta:0':
                val = pretrained_var_name['beta:0']
                print('m:0', scope_name, val.value.shape, var.shape)

            elif graph_var_name == 'gamma:0':
                val = pretrained_var_name['gamma:0']
                print('m:0', scope_name, val.value.shape, var.shape)

            else:
                print ('############### ', graph_var_name)
                val = None
                var = None
                
            if val.value.shape != var.shape:
                raise ValueError('Mismatch is shape of pretrained weights and network defined weights')
            

                
            # for keyy, vv in pretrained_weights[scope_name].items():
            #     print (keyy, vv)
        except KeyError:
            print('OOPS not found for ', scope_name)


def load_weights(filepath, by_name=False, exclude=None):
    """

    """
    import h5py
    from keras.engine import topology
    
    if exclude:
        by_name = True
    
    if h5py is None:
        raise ImportError('load_weights requires h5py.')
    
    f = h5py.File(filepath, mode='r')
    # isFile = isinstance(f, h5py.File)
    # isGroup = isinstance(f, h5py.Group)
    # isDataset = isinstance(f, h5py.Dataset)
    
    # print ([i for i in f.attrs])
    # print(isFile, isGroup, isDataset)
    # if 'layer_names' not in f.attrs and 'model_weights' in f:
    #     f = f['model_weights']
    for key, value in f.items():
        isFile = isinstance(value, h5py.File)
        isGroup = isinstance(value, h5py.Group)
        isDataset = isinstance(value, h5py.Dataset)
        # print (isFile, isGroup, isDataset)
        # print ('sadasdasdasdasd')
        if isGroup:
            for key2, value2 in value.items():
                isFile = isinstance(value2, h5py.File)
                isGroup = isinstance(value2, h5py.Group)
                isDataset = isinstance(value2, h5py.Dataset)
                for key3, value3 in value2.items():
                    if key != key2:
                        print('PROBLEM PROBLEM PROBLEM')
                    print(key, key2, key3, value3.shape)
        elif isDataset:
            print('Dataset ...........')
        elif isFile:
            print('File ...........')
            # break
            
            # keras_model = self.keras_model
            # layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            #     else keras_model.layers
            #
            # # Exclude some layers
            # if exclude:
            #     layers = filter(lambda l: l.name not in exclude, layers)
            #
            # if by_name:
            #     topology.load_weights_from_hdf5_group_by_name(f, layers)
            # else:
            #     topology.load_weights_from_hdf5_group(f, layers)
            # if hasattr(f, 'close'):
            #     f.close()
            #
            # # Update the log directory
            # self.set_log_dir(filepath)


# filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
# load_weights(filepath, by_name=False, exclude=None)