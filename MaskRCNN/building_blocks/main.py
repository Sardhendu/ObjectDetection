
import logging
import numpy as np
import tensorflow as tf
from MaskRCNN.config import config as conf
from MaskRCNN.building_blocks import load_params
from MaskRCNN.building_blocks.fpn import fpn_bottom_up_graph, fpn_top_down_graph
from MaskRCNN.building_blocks.rpn import rpn_graph
from MaskRCNN.building_blocks.proposals import ProposalLayer

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")



def inference():
    xIN = tf.placeholder(dtype=tf.float32,
                         shape=[None] + conf.IMAGE_SHAPE,
                         name='input_image')
    
    
    input_comp_graph = dict(xIN=xIN)
    
    C2, C3, C4, C5 = fpn_bottom_up_graph(xIN, 'resnet101') # Basically the Resnet architecture.
    
    # CREATE THE FPN GRAPH
    fpn_comp_graph = fpn_top_down_graph(C2, C3, C4, C5) # The Fractionally strided convolutions
    
    # CREATE THE RPN GRAPH
    rpn_comp_graph = rpn_graph(depth=256)
    
    # CREATE THE PROPOSAL GRAPH
    obj_pl = ProposalLayer(conf)
    proposal_graph = obj_pl.proposals(inference_batch_size=3)
    # proposal_graph = dict(rpn_probs=rpn_probs, rpn_box=rpn_box, input_anchors=input_anchors, )

    # Here run the session for rpn_graph feed it with input P2, P3, P4 P5,
    # Record the outputs in a numpy array and perform the proposals
    
    return input_comp_graph, fpn_comp_graph, rpn_comp_graph, proposal_graph



def main(pretrained_weights_path):
    print('')
    input_comp_graph, fpn_comp_graph, rpn_comp_graph, proposal_graph = inference()
    
    feed_dict = {input_comp_graph['xIN']: np.random.random((1, 1024, 1024, 3))}
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        # Note setting the weight can take 1-2 min due to the deep network
        # load_params.set_pretrained_weights(sess, pretrained_weights_path)
        
        p2, p3, p4, p5 = sess.run([fpn_comp_graph['fpn_p2'], fpn_comp_graph['fpn_p3'],
                                   fpn_comp_graph['fpn_p4'], fpn_comp_graph['fpn_p5']], feed_dict=feed_dict)
        
        print(p2.shape, p3.shape, p4.shape, p5.shape)

        rpn_logits = []
        rpn_probs = []
        rpn_bboxes = []
        for fpn_p in [p2, p3, p4, p5]:
            rpn_logit, rpn_prob, rpn_bbox = sess.run([rpn_comp_graph['rpn_class_logits'],
                                                      rpn_comp_graph['rpn_probs'],
                                                      rpn_comp_graph['rpn_bbox']],
                                                     feed_dict={rpn_comp_graph['xrpn']:fpn_p})
            rpn_logits.append(rpn_logit)
            rpn_probs.append(rpn_prob)
            rpn_bboxes.append(rpn_bbox)
            print(rpn_logit.shape, rpn_prob.shape, rpn_bbox.shape)
            del rpn_logit
            del rpn_prob
            del rpn_bbox
        
        
            
        print(len(rpn_logits), len(rpn_probs), len(rpn_bboxes))

        # Concatenate with the second dimension
        rpn_logits = np.concatenate(rpn_logits, axis=1)
        rpn_probs = np.concatenate(rpn_probs, axis=1)
        rpn_bbox = np.concatenate(rpn_probs, axis=1)
        
        
        proposals_ = sess.run([proposal_graph['proposals']],
                              feed_dict={
                                  proposal_graph['rpn_probs']: rpn_probs,
                                  proposal_graph['rpn_bbox']: rpn_bbox,
                                  
                              })

        # print(rpn_logits.shape)

    return p2, p3, p4, p5


filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
main(filepath)


# filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
# load_weights(filepath, by_name=False, exclude=None)