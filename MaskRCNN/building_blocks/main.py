
import logging
import numpy as np
import tensorflow as tf
from MaskRCNN.config import config as conf
from MaskRCNN.building_blocks import load_params
from MaskRCNN.building_blocks.fpn import fpn_bottom_up_graph, fpn_top_down_graph
from MaskRCNN.building_blocks.rpn import rpn_graph

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")



def inference():
    xIN = tf.placeholder(dtype=tf.float32,
                         shape=[None] + conf.IMAGE_SHAPE,
                         name='input_image')
    
    input_anchors = tf.placeholder(dtype=tf.float32,
                                   shape=[None, None, 4],
                                   name="input_anchors")
    
    C2, C3, C4, C5 = fpn_bottom_up_graph(xIN, 'resnet101') # Basically the Resnet architecture.
    #
    P2, P3, P4, P5 = fpn_top_down_graph(C2, C3, C4, C5) # The Fractionally strided convolutions
    # rpn_ = rpn_graph(depth=256)
    
    
    
    # Here run the session for rpn_graph feed it with input P2, P3, P4 P5,
    # Record the outputs in a numpy array and perform the proposals
    
    return dict(
            xIN=xIN,
            input_anchors=input_anchors,
            fpn_C2=C2, fpn_C3=C3, fpn_C4=C4, fpn_C5=C5,
            fpn_P2=P2, fpn_P3=P3, fpn_P4=P4, fpn_P5=P5
    )



def main(pretrained_weights_path):
    print('')
    inference_graph = inference()
    
    feed_dict = {inference_graph['xIN']: np.random.random((1, 1024, 1024, 3))}
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        # Note setting the weight can take 1-2 min due to the deep network
        # load_params.set_pretrained_weights(sess, pretrained_weights_path)
        
        c2, c3, c4, c5, p2, p3, p4, p5 = sess.run([inference_graph['fpn_C2'], inference_graph['fpn_C3'],
                                                    inference_graph['fpn_C4'], inference_graph['fpn_C5'],
                                                    inference_graph['fpn_P2'], inference_graph['fpn_P3'],
                                                    inference_graph['fpn_P4'], inference_graph['fpn_P5']
                                                  ], feed_dict=feed_dict)
        
        print(c2.shape, c3.shape, c4.shape, c5.shape, p2.shape, p3.shape, p4.shape, p5.shape)
    return c2, c3, c4, c5, p2, p3, p4, p5


# filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
# main(filepath)


# filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
# load_weights(filepath, by_name=False, exclude=None)