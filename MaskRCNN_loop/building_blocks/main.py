
import os
import logging

import numpy as np
from scipy import ndimage
from scipy import misc
import tensorflow as tf

from MaskRCNN_loop.config import config as conf
from MaskRCNN_loop.building_blocks import preprocess
from MaskRCNN_loop.building_blocks import load_params
from MaskRCNN_loop.building_blocks.fpn import FPN
from MaskRCNN_loop.building_blocks.rpn import RPN
from MaskRCNN_loop.building_blocks.proposals import Proposals
from MaskRCNN_loop.building_blocks.maskrcnn import MaskRCNN
from MaskRCNN_loop.building_blocks.detection import DetectionLayer
from MaskRCNN_loop.building_blocks import utils


logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def get_trainable_variable_name(sess):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print ("Variable name: Shape: ", k, v.shape)

# https://github.com/CharlesShang/FastMaskRCNN

def inference(image_shape, inference_batch_size, image_meta):
    xIN = tf.placeholder(dtype=tf.float32,
                         shape=[None] + conf.IMAGE_SHAPE,
                         name='input_image')
    
    input_anchors = tf.placeholder(dtype=tf.float32,
                         shape=[None, None, 4],
                         name='input_anchors')
    # CRATE THE FPN GRAPH
    feature_maps = FPN(xIN, 'resnet101').get_feature_maps() # Basically the Resnet architecture.

    # CREATE THE RPN GRAPH
    rpn_class_probs = []
    rpn_bbox = []
    for fmaps in feature_maps:
        obj_rpn = RPN(depth=256, feature_map=fmaps)
        print (obj_rpn.get_rpn_class_probs().shape, obj_rpn.get_rpn_bbox().shape)
        rpn_class_probs.append(obj_rpn.get_rpn_class_probs())
        rpn_bbox.append(obj_rpn.get_rpn_bbox())

    rpn_class_probs = tf.concat(rpn_class_probs, axis=1)
    rpn_bbox = tf.concat(rpn_bbox, axis=1)


    # # CREATE THE PROPOSAL GRAPH
    proposals = Proposals(conf, batch_size=inference_batch_size, rpn_class_probs=rpn_class_probs,
                          rpn_bbox=rpn_bbox, input_anchors=input_anchors, run_batch=False, DEBUG=False).get_proposals()
    #
    # # CREATE THE MRCNN GRAPH
    obj_mrcnn = MaskRCNN(image_shape=[1024, 1024, 3], pool_shape=[7, 7], num_classes=81, levels=[2, 3, 4, 5],
                         feature_maps=feature_maps[0:-1], proposals=proposals, type='keras', DEBUG=False)
    mrcnn_class_probs = obj_mrcnn.get_mrcnn_class_probs()
    mrcnn_bbox = obj_mrcnn.get_mrcnn_bbox()
    
    #
    # # DETECTION GRAPH
    # detections_grpah = DetectionLayer(conf, image_shape=[1024, 1024, 3], num_batches=inference_batch_size,
    #                             window=np.array(image_meta[:,7:11], dtype='int32'),
    #                             proposals=proposal_graph['proposals'],
    #                             mrcnn_class_probs=mrcnn_graph['mrcnn_class_probs'],
    #                             mrcnn_bbox=mrcnn_graph['mrcnn_bbox'], DEBUG=False).get_detections()
    #
    return xIN, input_anchors, mrcnn_class_probs, mrcnn_bbox # feature_maps, rpn_class_probs, rpn_bbox, proposals,



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
    print(image_metas)

    # Built the computation graph
    # image_shape = transformed_images[0].shape#[1024,1024,3]
    batch_size = transformed_images.shape[0]
    xIN, input_anchors, mrcnn_class_probs, mrcnn_bbox = inference(image_shape, batch_size, image_metas)

    init = tf.global_variables_initializer()


    DEBUG = False
    with tf.Session() as sess:
        sess.run(init)

        if DEBUG:
            # PRINT ALL THE TRAINABLE VARIABLES
            get_trainable_variable_name(sess)
            load_params.set_pretrained_weights(sess, pretrained_weights_path)
        else:
            # Get input Image:
            # Note setting the weight can take 1-2 min due to the deep network
            # load_params.set_pretrained_weights(sess, pretrained_weights_path)

            # RUN FPN GRAPH
            resnet_stage_shapes = utils.get_resnet_stage_shapes(conf, image_shape=image_shape)
            anchors = utils.gen_anchors(image_shape=image_shape,
                                        batch_size=batch_size,
                                        scales=conf.RPN_ANCHOR_SCALES,
                                        ratios=conf.RPN_ANCHOR_RATIOS,
                                        feature_map_shapes=resnet_stage_shapes,
                                        feature_map_strides=conf.RESNET_STRIDES,
                                        anchor_strides=conf.RPN_ANCHOR_STRIDE)
            
            
            # feed_dict = {xIN: transformed_images, input_anchors: anchors}#np.random.random((batch_size, 1024, 1024, 3))}
            # feature_maps_, rpn_class_probs_, rpn_bbox_, proposals_ = sess.run([
            #     feature_maps, rpn_class_probs, rpn_bbox, proposals], feed_dict=feed_dict)
            # print('(ANCHORS): ', anchors.shape)
            # print('(FPN) feature_maps_: ', len(feature_maps_))
            # print('(RPN) rpn_class_probs_: ', rpn_class_probs_.shape)
            # print('(RPN) rpn_bbox_: ', rpn_bbox_.shape)
            # print('(PROPOSALS) proposals_: ', proposals_.shape)

            feed_dict = {xIN: transformed_images, input_anchors: anchors}
            mrcnn_class_probs_, mrcnn_bbox_ = sess.run([mrcnn_class_probs, mrcnn_bbox], feed_dict=feed_dict)
            print('(MRCNN) mrcnn_class_probs_: ', mrcnn_class_probs_.shape)
            print('(MRCNN) mrcnn_bbox_: ', mrcnn_bbox_.shape)
    #
    # print (sess)
    # with  tf.Session() as sess1:
    #     sess.init()

  
  
  
  
  
  
    #         # RUN RPN GRAPH
    #         # rpn_logits = []
    #         print('RUNNING RPN ..............')
    #         rpn_probs = []
    #         rpn_bboxes = []
    #         for fpn_p in [p2, p3, p4, p5, p6]:
    #             _, rpn_prob, rpn_bbox = sess.run([rpn_comp_graph['rpn_class_logits'],
    #                                                       rpn_comp_graph['rpn_class_probs'],
    #                                                       rpn_comp_graph['rpn_bbox']],
    #                                                      feed_dict={rpn_comp_graph['xrpn']:fpn_p})
    #             # rpn_logits.append(rpn_logit)
    #             rpn_probs.append(rpn_prob)
    #             rpn_bboxes.append(rpn_bbox)
    #             print('RPN: rpn_class_score=%s, rpn_bbox=%s '%(str(rpn_prob.shape), str(rpn_bbox.shape)))
    #             # del rpn_logit
    #             del rpn_prob
    #             del rpn_bbox
    #
    #         # Concatenate with the second dimension
    #         # rpn_logits = np.concatenate(rpn_logits, axis=1)
    #         rpn_probs = np.concatenate(rpn_probs, axis=1)
    #         rpn_bboxes = np.concatenate(rpn_bboxes, axis=1)
    #         print('RPN Total(stacked): rpn_class_score=%s, rpn_bbox=%s ' % (str(rpn_probs.shape), str(rpn_bboxes.shape)))
    #
    #         # RUN PROPOSAL GRAPH AND THE MASK-RCNN GRAPH AND DETECTION TOGETHER
    #         # Get Anchors
    #         print('GENERATING ANCHORS ..............')
    
    #
    #         print('RUNNING PROPOSAL AND MRCNN ..............')
    #         _, proposals_, detections_ = sess.run(
    #                 [proposal_graph['proposals'], detections],
    #                                feed_dict={
    #                                    proposal_graph['rpn_probs']: rpn_probs,
    #                                    proposal_graph['rpn_bbox']: rpn_bboxes,
    #                                    proposal_graph['input_anchors']: anchors,
    #                                    mrcnn_graph['P2']: p2, mrcnn_graph['P3']: p3,
    #                                    mrcnn_graph['P4']: p4, mrcnn_graph['P5']: p5
    #                                 })
    #         print('(PROPOSALS) Generated output shape: ', proposals_.shape)
    #         # print('(MRCNN) mrcnn_class_probs: ', mrcnn_class_probs_.shape)
    #         # print('(MRCNN) mrcnn_bbox: ', mrcnn_bbox_.shape)
    #         # print('(DETECTION) detection: ', detections_.shape)






filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
main(filepath)


# filepath = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
# load_weights(filepath, by_name=False, exclude=None)






# # MRCNN LAYER: Classify the object and get the bounding box
# feed_dict = feed_dict = {p_graph['rpn_probs']: a, p_graph['rpn_bbox']: b, p_graph['input_anchors']: c,
#      mrcnn_graph['P2']:P2, mrcnn_graph['P3']:P3, mrcnn_graph['P4']:P4, mrcnn_graph['P5']:P5 }
# mrcnn_class_probs_, mrcnn_bbox_ = sess.run(mrcnn_graph['pooled_rois'], mrcnn_graph['mrcnn_class_probs'],
#                                    mrcnn_graph['mrcnn_bbox'])






# ROUGH
#
# import tensorflow as tf
# def conv_layer(X, k_shape, seed, scope_name='conv_layer'):
#     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
#         weight = tf.get_variable(
#                 dtype=tf.float32,
#                 shape=k_shape,
#                 initializer=tf.truncated_normal_initializer(
#                         stddev=0.1, seed=seed
#                 ),
#                 name="kernel",
#                 trainable=True
#         )
#
#
#     return tf.matmul(X, weight), weight
#
#
# import numpy as np
# np.random.seed(343)
# x = np.array(np.random.random((2, 2,2)), dtype='float32')
# print(x)
# out ,wght= conv_layer(X=x[0], k_shape=[2,2], scope_name='conv_layer', seed=13)
# out2 ,wght2= conv_layer(X=x[1], k_shape=[2,2], scope_name='conv_layer1', seed=15)
# print (out)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     out_, wght_, out2_, wght2_ = sess.run([out,wght, out2, wght2])
#     print(wght_)
#     print ('')
#     print(wght2_)
#     print('')
#     print(out_)
#     print('')
#     print(out2_)
#
#
    
#
# [[[ 0.07505603  0.64601904]
#   [ 0.65630102  0.42003104]]
#
#  [[ 0.23901746  0.92122155]
#   [ 0.9318316   0.10857289]]]

#
# wght_
# [[ 0.01745166 -0.0225782 ]
#  [ 0.01967431  0.0387477 ]]
#
# wght2_
# [[ 0.01745166 -0.0225782 ]
#  [ 0.01967431  0.0387477 ]]
#
# out_
# [[ 0.01401983  0.02333712]
#  [ 0.01971737  0.00145714]]
#
# out2_
# [[ 0.02229565  0.03029863]
#  [ 0.01839811 -0.01683213]]