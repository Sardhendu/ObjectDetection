
import logging
import tensorflow as tf
from MaskRCNN.config import config as conf
from MaskRCNN.building_blocks import load_params
from MaskRCNN.building_blocks.fpn import fpn_bottom_up_graph, fpn_top_down_graph
from MaskRCNN.building_blocks.rpn import rpn_graph

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")



def inference(pretrained_weights_path):
    xIN = tf.placeholder(dtype=tf.float32,
                         shape=[None] + conf.IMAGE_SHAPE,
                         name='input_image')
    
    input_anchors = tf.placeholder(dtype=tf.float32,
                                   shape=[None, None, 4],
                                   name="input_anchors")
    
    C2, C3, C4, C5 = fpn_bottom_up_graph(xIN, 'resnet101') # Basically the Resnet architecture.
    P2, P3, P4, P5 = fpn_top_down_graph(C2, C3, C4, C5) # The Fractionally strided convolutions
    rpn_ = rpn_graph(depth=256)
    load_params.check_params_consistency(pretrained_weights_path)
    
    
    # Here run the session for rpn_graph feed it with input P2, P3, P4 P5,
    # Record the outputs in a numpy array and perform the proposals






filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
inference(filepath)
# filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
# load_weights(filepath, by_name=False, exclude=None)