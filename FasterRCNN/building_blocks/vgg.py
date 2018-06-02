
import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


class vgg16():
    def __init__(self, mode='train', model_path=None):
        '''
        :param mode:
        :param model_path:
        
        Note: We don't make the use of Fully connected layers, Our aim is just to get the feature map
        on which we could perform Region Proposal (By the Use of RPN)
        '''
        self.mode = mode
        
        if mode == 'test':
            self.model = np.load(model_path, encoding="latin1")
        else:
            self.model = None
        
        # print (self.model.keys())
    
    def init_weights(self, shape=None, name='w'):
        if self.mode == 'train':
            init_ =tf.truncated_normal_initializer(stddev=0.001)
            name_ = name
            shape_ = shape
        else:
            w_tensor = self.model[name]
            init_ = tf.constant_initializer(value=w_tensor, dtype=tf.float32)
            shape_ = w_tensor.shape
            name_ = name
        
        w = tf.get_variable(initializer=init_, shape=shape_, name=name_)
        return w
    
    def init_bias(self, shape=None, name='b'):
        if self.mode == 'train':
            init_ = tf.constant_initializer(1)
            shape_ = shape
            name_ = name
        else:
            b_tensor = self.model[name]
            # print(b_tensor)
            init_ = tf.constant_initializer(value=b_tensor, dtype=tf.float32)
            shape_ = b_tensor.shape
            name_ = name + "Bias"
            
        b =  tf.get_variable(initializer=init_, shape=shape_, name=name_)
        return b

    def conv2d(self, x, w_shape, b_shape, scope_name):
        
        with tf.variable_scope(scope_name):
            W = self.init_weights(w_shape, scope_name + '_W')
            b = self.init_bias(b_shape, scope_name + '_b')
            x = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            x = tf.nn.relu(x)
            logging.info('%s shape = %s', str(scope_name), x.get_shape().as_list())
            return x
        
    def max_pool(self, x, scope_name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
    def get_feature_map(self, image_shape):
        xIN = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]])
        # print(x.get_shape().as_list())
        x = self.conv2d(xIN,  [3, 3, 3, 64],  [64], scope_name="conv1_1")
        x = self.conv2d(x,  [3, 3, 64, 64], [64], scope_name="conv1_2")
        x = self.max_pool(x, scope_name="pool1")

        x = self.conv2d(x,   [3, 3, 64, 128],   [128], scope_name="conv2_1")
        x = self.conv2d(x,   [3, 3, 128, 128],  [128], scope_name="conv2_2")
        x = self.max_pool(x, scope_name="pool2")

        x = self.conv2d(x,   [3, 3, 128, 256],  [256], scope_name="conv3_1")
        x = self.conv2d(x,   [3, 3, 256, 256],  [256], scope_name="conv3_2")
        x = self.conv2d(x,   [3, 3, 256, 256],  [256], scope_name="conv3_3")
        x = self.max_pool(x, scope_name="pool3")

        x = self.conv2d(x,   [3, 3, 256, 512],  [512], scope_name="conv4_1")
        x = self.conv2d(x,   [3, 3, 512, 512],  [512], scope_name= "conv4_2")
        x = self.conv2d(x,   [3, 3, 512, 512],  [512], scope_name="conv4_3")
        x = self.max_pool(x, scope_name="pool4")

        x = self.conv2d(x,   [3, 3, 512, 512],  [512], scope_name="conv5_1")
        x = self.conv2d(x,   [3, 3, 512, 512],  [512], scope_name="conv5_2")
        x = self.conv2d(x,   [3, 3, 512, 512],  [512], scope_name="conv5_3")
        
        return xIN, x


def debugg():
    model_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/FasterRCNN/VGG_imagenet.npy'
    obj_vgg = vgg16(mode='test', model_path=model_path)
    input_image, feature_map = obj_vgg.get_feature_map([224, 224, 3])
    print(feature_map.shape)
    
    # LOAD MODEL AND TEST
    # model_2path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/FasterRCNN/vgg16_weights.npz'
    # model = np.load(model_path, encoding="latin1").item()
    # model2 = np.load(model_2path, encoding="latin1")
    # # print (model)
    # for k, v in model.items():
    #     if k == 'conv1_1':
    #         print (v['biases'])#, v.shape)#['conv4_3_W'].shape)
    #
    # for k, v in model2.items():
    #     if k == 'conv1_1_b':
    #         print (v)

# debugg()

