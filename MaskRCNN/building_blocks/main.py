
import os
import logging

import numpy as np
from scipy import ndimage
from scipy import misc
import tensorflow as tf

from MaskRCNN.config import config as conf
from MaskRCNN.building_blocks import preprocess
from MaskRCNN.building_blocks import load_params
from MaskRCNN.building_blocks.fpn import FPN
from MaskRCNN.building_blocks.rpn import RPN
from MaskRCNN.building_blocks.proposals_tf import Proposals
from MaskRCNN.building_blocks import utils

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def get_trainable_variable_name(sess):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print ("Variable name: Shape: ", k, v.shape)

# https://github.com/CharlesShang/FastMaskRCNN

def inference(inference_batch_size):
    xIN = tf.placeholder(dtype=tf.float32,
                         shape=[None] + conf.IMAGE_SHAPE,
                         name='input_image')
    
    
    input_comp_graph = dict(xIN=xIN)
    
    # CRATE THE FPN GRAPH
    fpn_graph = FPN(xIN, 'resnet101').get_fpn_graph() # Basically the Resnet architecture.
    
    # CREATE THE RPN GRAPH
    rpn_comp_graph = RPN(depth=256).get_rpn_graph()
    
    # CREATE THE PROPOSAL GRAPH
    proposal_graph = Proposals(conf, batch_size=inference_batch_size).get_proposal_graph()
    
    # Here run the session for rpn_graph feed it with input P2, P3, P4 P5,
    # Record the outputs in a numpy array and perform the proposals
    
    return input_comp_graph, fpn_graph, rpn_comp_graph, proposal_graph



def main(pretrained_weights_path):
    print('')
    # Get Images
    img_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/images/3627527276_6fe8cd9bfe_z.jpg'
    image = ndimage.imread(img_path, mode='RGB')
    image_id = os.path.basename(img_path).split('.')[0]
    
    ## Process Images:
    list_of_images = [image]
    list_of_image_ids = [image_id]
    transformed_images, image_metas, image_windows, anchors = preprocess.process_images(list_of_images, list_of_image_ids)
    image_shape = transformed_images.shape[1:]
    print ('Shape of input image batch: ', image_shape)
    
    # Built the computation graph
    # image_shape = transformed_images[0].shape#[1024,1024,3]
    batch_size = transformed_images.shape[0]
    input_comp_graph, fpn_comp_graph, rpn_comp_graph, proposal_graph = inference(batch_size)


    init = tf.global_variables_initializer()
    
    
    DEBUG = True
    with tf.Session() as sess:
        sess.run(init)

        if DEBUG:
            # PRINT ALL THE TRAINABLE VARIALES
            get_trainable_variable_name(sess)
        else:
            # Get input Image:
            # Note setting the weight can take 1-2 min due to the deep network
            # load_params.set_pretrained_weights(sess, pretrained_weights_path)
    
            # RUN FPN GRAPH
            feed_dict = {input_comp_graph['xIN']: transformed_images}#np.random.random((batch_size, 1024, 1024, 3))}
            p2, p3, p4, p5, p6 = sess.run([fpn_comp_graph['fpn_p2'], fpn_comp_graph['fpn_p3'],
                                           fpn_comp_graph['fpn_p4'], fpn_comp_graph['fpn_p5'],
                                           fpn_comp_graph['fpn_p6']], feed_dict=feed_dict)
        #
            print('FPN P2=%s, P3=%s, P4=%s, P5=%s, P6=%s'%(str(p2.shape), str(p3.shape), str(p4.shape), str(p5.shape), str(p6.shape)))
    
            # RUN RPN GRAPH
            # rpn_logits = []
            rpn_probs = []
            rpn_bboxes = []
            for fpn_p in [p2, p3, p4, p5, p6]:
                _, rpn_prob, rpn_bbox = sess.run([rpn_comp_graph['rpn_class_logits'],
                                                          rpn_comp_graph['rpn_class_probs'],
                                                          rpn_comp_graph['rpn_bbox']],
                                                         feed_dict={rpn_comp_graph['xrpn']:fpn_p})
                # rpn_logits.append(rpn_logit)
                rpn_probs.append(rpn_prob)
                rpn_bboxes.append(rpn_bbox)
                print('RPN: rpn_class_score=%s, rpn_bbox=%s '%(str(rpn_prob.shape), str(rpn_bbox.shape)))
                # del rpn_logit
                del rpn_prob
                del rpn_bbox
    
            # Concatenate with the second dimension
            # rpn_logits = np.concatenate(rpn_logits, axis=1)
            rpn_probs = np.concatenate(rpn_probs, axis=1)
            rpn_bboxes = np.concatenate(rpn_bboxes, axis=1)
            print('RPN Total(stacked): rpn_class_score=%s, rpn_bbox=%s ' % (str(rpn_probs.shape), str(rpn_bboxes.shape)))
    
            # RUN PROPOSAL GRAPH
            # Get Anchors
            resnet_stage_shapes = utils.get_resnet_stage_shapes(conf, image_shape=image_shape)
            anchors = utils.gen_anchors(image_shape=[1024, 1024, 3],
                                  batch_size=2, scales=conf.RPN_ANCHOR_SCALES,
                                  ratios=conf.RPN_ANCHOR_RATIOS,
                                  feature_map_shapes=resnet_stage_shapes,
                                  feature_map_strides=conf.RESNET_STRIDES,
                                  anchor_strides=conf.RPN_ANCHOR_STRIDE)
    
            proposals_ = sess.run(proposal_graph['proposals'],
                                   feed_dict={
                                      proposal_graph['rpn_probs']: rpn_probs,
                                      proposal_graph['rpn_bbox']: rpn_bboxes,
                                      proposal_graph['input_anchors']: anchors
    
                                  })
    
            print('(PROPOSALS) Generated output shape: ', proposals_.shape)
            print ('')
            print(proposals_)

    # return p2, p3, p4, p5

#
filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
main(filepath)


# filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
# load_weights(filepath, by_name=False, exclude=None)