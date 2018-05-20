

import logging
import tensorflow as tf
from MaskRCNN.config import config as conf

from MaskRCNN.building_blocks.fpn import fpn_bottom_up_graph, fpn_top_down_graph
from MaskRCNN.building_blocks.rpn import rpn_graph

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def main():
    xIN = tf.placeholder(dtype=tf.float32,
                         shape=[None] + conf.IMAGE_SHAPE,
                         name='input_image')
    
    C2, C3, C4, C5 = fpn_bottom_up_graph(xIN) # Basically the Resnet architecture.
    P2, P3, P4, P5 = fpn_top_down_graph(C2, C3, C4, C5) # The Fractionally strided convolutions
    rpn_ = rpn_graph(depth=256)

main()