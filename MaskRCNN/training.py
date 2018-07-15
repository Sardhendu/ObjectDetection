
import os
import logging

import pickle
import numpy as np
import h5py
from scipy import ndimage
from scipy import misc
import tensorflow as tf

# from MaskRCNN.config import config as conf
from MaskRCNN.building_blocks import data_processor
from MaskRCNN.building_blocks import load_params
from MaskRCNN.building_blocks.fpn import FPN
from MaskRCNN.building_blocks.rpn import RPN
from MaskRCNN.building_blocks.proposals_tf import Proposals
from MaskRCNN.building_blocks.maskrcnn import MaskRCNN
from MaskRCNN.building_blocks.loss_optimize import Loss
from MaskRCNN.building_blocks.detection import DetectionLayer, unmold_detection
from MaskRCNN.building_blocks import utils

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")





# TODO: First build a training model
# TODO: Fix some layers, (Train only the RPN and MRCNN head)
# TODO: Get data and run it through RPN check if the output shapes match
# TODO: Make a complete run train for few iteration.


class Train():
    def __init__(self, conf, batch_size, pretrained_weights_path):
        self.batch_size = batch_size
        self.pretrained_weights_path = pretrained_weights_path
        self.conf = conf
        print (self.conf)
    
    def build(self):
        pass
    
    def transform_images(self, data_dict, image_ids):
        batch_images = data_dict['batch_images']
        batch_gt_masks = data_dict['batch_gt_masks']
        batch_gt_class_ids = data_dict['batch_gt_class_ids']
        batch_gt_bboxes = data_dict['batch_gt_bboxes']
        batch_image_metas = data_dict['batch_image_metas']
        batch_rpn_target_class = data_dict['batch_rpn_target_class']
        batch_rpn_target_bbox = data_dict['batch_rpn_target_bbox']
        # print (batch_image_metas)
        transformed_images, image_metas, image_windows, anchors = data_processor.process_images(self.conf,
                                                                                                batch_images, image_ids)
        
        print (transformed_images.shape, image_metas.shape, image_windows.shape,
               anchors.shape)
        
        return (batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes, batch_image_metas, batch_rpn_target_class,
        batch_rpn_target_bbox, anchors)
    
    def get_network_inputs(self):
        ''' NOTE

        The gt_class_ids, gt_bboxes, are zero padded. This means that we may have actually only found 1 gt_bbox and 1
        gt_class_ids, and rest 99 are just added to stack them
        :return:
        '''
        xIN = tf.placeholder(dtype=tf.float32,
                             shape=[None] + self.conf.IMAGE_SHAPE,
                             name='input_image')
        
        gt_masks = tf.placeholder(dtype=tf.float32,
                                  shape=[None] + self.conf.IMAGE_SHAPE[:2] + [self.conf.MAX_GT_OBJECTS],
                                  name='batch_gt_masks')
        gt_class_ids = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.conf.MAX_GT_OBJECTS],
                                      name='gt_class_ids')
        
        gt_bboxes = tf.placeholder(dtype=tf.float32,
                                   shape=[None, self.conf.MAX_GT_OBJECTS, 4],
                                   name='gt_bboxes')
        
        rpn_target_class = tf.placeholder(dtype=tf.float32,
                                          shape=[None, None, 1],
                                          name='rpn_target_class')
        
        rpn_target_bbox = tf.placeholder(dtype=tf.float32,
                                         shape=[None, None, 4],
                                         name='rpn_target_bbox')
        
        anchors = tf.placeholder(dtype=tf.float32, shape=[None, None, 4],
                                 name="input_anchors")
        
        return xIN, gt_masks, gt_class_ids, gt_bboxes, rpn_target_class, rpn_target_bbox, anchors
    
    def build_train_graph(self):
        xIN, gt_masks, gt_class_ids, gt_bboxes, rpn_target_class, rpn_target_bbox, anchors = self.get_network_inputs()
        
        # CREATE TEH FPN GRAPH
        fpn_graph = FPN(self.conf, xIN, 'resnet101').get_fpn_graph()
        
        #
        # CREATE THE RPN GRAPH
        rpn_pred_logits = []
        rpn_pred_probs = []
        rpn_pred_bbox = []
        for fmap in [fpn_graph['fpn_p2'], fpn_graph['fpn_p3'], fpn_graph['fpn_p4'], fpn_graph['fpn_p5'],
                     fpn_graph['fpn_p6']]:
            rpn_obj = RPN(self.conf, depth=256, feature_map=fmap)  #
            rpn_pred_logits.append(rpn_obj.get_rpn_class_logits())
            rpn_pred_probs.append(rpn_obj.get_rpn_class_probs())
            rpn_pred_bbox.append(rpn_obj.get_rpn_bbox())

        rpn_pred_logits = tf.concat(rpn_pred_logits, axis=1)
        rpn_pred_probs = tf.concat(rpn_pred_probs, axis=1)
        rpn_pred_bbox = tf.concat(rpn_pred_bbox, axis=1)
        
        # CREATE THE PROPOSAL GRAPH
        proposals = Proposals(self.conf, self.batch_size,
                              rpn_pred_probs, rpn_pred_bbox, anchors,
                              DEBUG=True).get_proposals()
        
        # MRCNN GRAPH
        mrcnn_graph = MaskRCNN(self.conf.IMAGE_SHAPE, pool_shape=[7, 7],
                               num_classes=self.conf.NUM_CLASSES,
                               levels=[2, 3, 4, 5], proposals=proposals,
                               feature_maps=[fpn_graph['fpn_p2'], fpn_graph['fpn_p3'],
                                             fpn_graph['fpn_p4'], fpn_graph['fpn_p5']],
                               type='keras', DEBUG=False).get_mrcnn_graph()
        
        # TODO: DETECTION LAYER
        # RPN has two losses 1) Classification loss and 2) Regularization
        
        # TODO: Create RPN LOSS
        # RPN has two losses 1) Classification loss and 2) Regularization
        rpn_class_loss = Loss.rpn_class_loss(rpn_target_class, rpn_pred_logits)
        rpn_target_bbox_nopad, rpn_pred_box_pos = Loss.rpn_box_loss(rpn_target_bbox, rpn_pred_bbox, rpn_target_class, batch_size=1)
        
        # TODO: DETECTION
        
        # TODO: Create MRCNN Loss  (Hmm how would we do it, when we havent compute the ground truth)
        
        return (fpn_graph, rpn_pred_logits, rpn_pred_probs, rpn_pred_bbox, proposals,
                mrcnn_graph, xIN, anchors, rpn_target_class, rpn_target_bbox,
                rpn_class_loss, rpn_target_bbox_nopad, rpn_pred_box_pos)
    
    def exec_sess(self, data_dict, image_ids):
        # TODO: Inputs anchors and xIN
        tf.reset_default_graph()
        
        # # BUILD THE GRAPH
        # (fpn_graph, rpn_pred_logits, rpn_pred_probs, rpn_pred_bbox, proposals,
        #         mrcnn_graph, xIN, anchors, rpn_target_class, rpn_target_bbox,
        #         rpn_class_loss, rpn_target_bbox_nopad, rpn_pred_box_pos) = self.build_train_graph()
        
        # GET INPUT DATA
        batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes, batch_image_metas, batch_rpn_target_class, \
        batch_rpn_target_bbox, anchors_ = self.transform_images(data_dict, image_ids)
        
        batch_active_class_ids = batch_image_metas[:,-4:]  # 1 corresponds to the active level
        
        print('batch_active_class_ids', batch_active_class_ids)
        print('batch_rpn_target_class ', batch_rpn_target_class.shape)
        
        
        
        
        
        
        ######### ROUGH ###########################
        from MaskRCNN.building_blocks import data_processor
        proposal__ = np.array([[[1, 10, 1, 10], [23, 54, 155, 177], [10,10,167,170],
                                [0, 0, 0, 0]],
                               [[3, 2, 2, 2], [54, 22, 144, 171], [0,0,0,0],[0, 0, 0, 0]]])
        
        # proposals_ = tf.placeholder(shape=(2, 3, 4), dtype=tf.float32, name='proposals')
        proposals_ = tf.placeholder(shape=(2, 4, 4), dtype=tf.float32, name='proposals')

        batch_rois = []
        batch_roi_gt_class_ids = []
        batch_roi_gt_class_boxes = []
        for i in range(0, 2):
            rois, roi_gt_class_ids, roi_gt_box_deltas = data_processor.BuildDetectionTargets(self.conf,
                    proposals_[i], batch_gt_bboxes[i], batch_gt_class_ids[i], DEBUG=False).get_target_rois()

            break
            # batch_proposals.append(prop)

       
            #
    
        ##########################################
        
        
        
        
        
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #
        #     # PRINT ALL THE TRAINING VARIABLES
        #     load_params.print_trainable_variable_names(sess)
        #
        #     # GET PRETRAINED WEIGHTS
        #     if self.pretrained_weights_path:
        #         load_params.set_pretrained_weights(sess, self.pretrained_weights_path,
        #                                            train_nets='heads')
        #
        #     (rpn_pred_logits_, rpn_pred_probs_, rpn_pred_bbox_, proposals_, mrcnn_class_logits_, mrcnn_class_probs_, mrcnn_bbox_) = sess.run([
        #                     rpn_pred_logits,
        #                     rpn_pred_probs,
        #                     rpn_pred_bbox,
        #                     proposals,
        #                     mrcnn_graph['mrcnn_class_logits'],
        #                     mrcnn_graph['mrcnn_class_probs'],
        #                     mrcnn_graph['mrcnn_bbox']],
        #             feed_dict={xIN: batch_images, anchors: anchors_})
        #
        #     print('Max and Min Proposals, ', np.amax(proposals_), np.amin(proposals_))
        #     print('Num NaN present in Proposals ', np.sum(np.isnan(proposals_)))
        #
        #     print('(MRCNN) proposals (shape) ', proposals_.shape)
        #
        #     # print(rpn_class_probs_.shape, rpn_bbox_.shape, mrcnn_class_probs_.shape, mrcnn_bbox_.shape)
        #
        #     print('rpn_pred_logits_.shape ', rpn_pred_logits_.shape)
        #     print('')
        #     print ('rpn_bbox_.shape ', rpn_pred_bbox_.shape)
        #     print('')
        #     print ('mrcnn_class_probs_.shape ', mrcnn_class_probs_.shape)
        #     print ('')
        #     print('mrcnn_bbox_.shape ', mrcnn_bbox_.shape)
        #
        #     rpn_class_loss_ = sess.run(rpn_class_loss,
        #                     feed_dict={
        #                         rpn_target_class: batch_rpn_target_class, rpn_pred_logits:rpn_pred_logits_
        #                     })
        #
        #     _,rpn_box_loss_ = sess.run([rpn_target_bbox_nopad, rpn_pred_box_pos],
        #                     feed_dict={
        #                         rpn_target_class: batch_rpn_target_class,
        #                         rpn_target_bbox: batch_rpn_target_bbox,
        #                         rpn_pred_bbox: rpn_pred_bbox_
        #                     })
        #
        #     print('mrcnn_class_logits_ ', mrcnn_class_logits_.shape, mrcnn_class_logits_)
        #
        #     print(rpn_class_loss_)
        #
        #     Loss.mrcnn_class_loss(mrcnn_target_class=batch_gt_class_ids,
        #                           mrcnn_pred_logits=mrcnn_class_logits_,
        #                           batch_active_class_ids=batch_active_class_ids,
        #                           sess=sess)
        #
        
        
        
        
        ##### ROUGH
            # print('')
            # print(rpn_box_loss_)
            # print('')
            # print(batch_rpn_target_bbox)
            # print('')


            # for i in range(0,10):
            #     print(batch_rpn_target_bbox[:,i,:])
            #
            # print('')
            # print('')
            # print('0 ', len(np.where(target_class_[0] == 0)[0]))
            # print('1 ', len(np.where(target_class_[0] == 1)[0]))
            # print('2 ', len(np.where(target_class_[0] == -1)[0]))
           
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            # import os
# import logging
#
# import pickle
# import numpy as np
# import h5py
# from scipy import ndimage
# from scipy import misc
# import tensorflow as tf
#
# # from MaskRCNN.config import config as conf
# from MaskRCNN.building_blocks import data_processor
# from MaskRCNN.building_blocks import load_params
# from MaskRCNN.building_blocks.fpn import FPN
# from MaskRCNN.building_blocks.rpn import RPN
# from MaskRCNN.building_blocks.proposals_tf import Proposals
# from MaskRCNN.building_blocks.maskrcnn import MaskRCNN
# from MaskRCNN.building_blocks.loss_optimize import Loss
# from MaskRCNN.building_blocks.detection import DetectionLayer, unmold_detection
# from MaskRCNN.building_blocks import utils
#
# logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
#                     format="%(asctime)-15s %(levelname)-8s %(message)s")
#
#
#
#
#
# # TODO: First build a training model
# # TODO: Fix some layers, (Train only the RPN and MRCNN head)
# # TODO: Get data and run it through RPN check if the output shapes match
# # TODO: Make a complete run train for few iteration.
#
#
# class Train():
#     def __init__(self, conf, batch_size):
#         self.batch_size = batch_size
#         self.conf = conf
#         print (self.conf)
#
#     def build(self):
#         pass
#
#     def transform_images(self, data_dict, image_ids):
#         batch_images = data_dict['batch_images']
#         batch_gt_masks = data_dict['batch_gt_masks']
#         batch_gt_class_ids = data_dict['batch_gt_class_ids']
#         batch_gt_bboxes = data_dict['batch_gt_bboxes']
#         batch_image_metas = data_dict['batch_image_metas']
#         batch_rpn_target_class = data_dict['batch_rpn_target_class']
#         batch_rpn_target_bbox = data_dict['batch_rpn_target_bbox']
#         # print (batch_image_metas)
#         transformed_images, image_metas, image_windows, anchors = data_processor.process_images(self.conf, batch_images, image_ids)
#
#         print (transformed_images.shape, image_metas.shape, image_windows.shape,
#                anchors.shape)
#
#         return (batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes, batch_image_metas, batch_rpn_target_class, batch_rpn_target_bbox, anchors)
#
#
#
#     def get_network_inputs(self):
#         ''' NOTE
#
#         The gt_class_ids, gt_bboxes, are zero padded. This means that we may have actually only found 1 gt_bbox and 1 gt_class_ids, and rest 99 are just added to stack them
#         :return:
#         '''
#         xIN = tf.placeholder(dtype=tf.float32,
#                              shape=[None] + self.conf.IMAGE_SHAPE,
#                              name='input_image')
#
#         gt_masks = tf.placeholder(dtype=tf.float32,
#                                   shape=[None] +  self.conf.IMAGE_SHAPE[:2]+ [self.conf.MAX_GT_OBJECTS],
#                                   name='batch_gt_masks')
#         gt_class_ids = tf.placeholder(dtype=tf.float32,
#                                       shape=[None, self.conf.MAX_GT_OBJECTS],
#                                       name='gt_class_ids')
#
#         gt_bboxes = tf.placeholder(dtype=tf.float32,
#                                    shape=[None, self.conf.MAX_GT_OBJECTS, 4],
#                                    name='gt_bboxes')
#
#         rpn_target_class = tf.placeholder(dtype=tf.float32,
#                                           shape=[None, None, 1],
#                                           name='rpn_target_class')
#
#         rpn_target_bbox = tf.placeholder(dtype=tf.float32,
#                                          shape=[None, None, 4],
#                                          name='rpn_target_bbox')
#
#         anchors = tf.placeholder(dtype=tf.float32, shape=[None, None, 4],
#                                  name="input_anchors")
#
#
#
#
#         return xIN, gt_masks, gt_class_ids, gt_bboxes, rpn_target_class, rpn_target_bbox, anchors
#
#
#     def build_train_graph(self):
#         xIN, gt_masks, gt_class_ids, gt_bboxes, rpn_target_class, rpn_target_bbox, anchors = self.get_network_inputs()
#
#         # CREATE TEH FPN GRAPH
#         fpn_graph = FPN(self.conf, xIN, 'resnet101').get_fpn_graph()
#
#         #
#         # CREATE THE RPN GRAPH
#         rpn_class_logits = []
#         rpn_class_probs = []
#         rpn_bbox = []
#         for fmap in [fpn_graph['fpn_p2'], fpn_graph['fpn_p3'], fpn_graph['fpn_p4'], fpn_graph['fpn_p5'], fpn_graph['fpn_p6']]:
#             rpn_obj = RPN(self.conf, depth=256, feature_map=fmap)#
#             rpn_class_logits.append(rpn_obj.get_rpn_class_logits())
#             rpn_class_probs.append(rpn_obj.get_rpn_class_probs())
#             rpn_bbox.append(rpn_obj.get_rpn_bbox())
#
#         rpn_class_logits = tf.concat(rpn_class_logits, axis=1)
#         rpn_class_probs = tf.concat(rpn_class_probs, axis=1)
#         rpn_bbox = tf.concat(rpn_bbox, axis=1)
#
#         # CREATE THE PROPOSAL GRAPH
#         proposals = Proposals(self.conf, self.batch_size, rpn_class_probs, rpn_bbox,  anchors,
#                               DEBUG=False).get_proposals()
#
#         # MRCNN GRAPH
#         mrcnn_graph = MaskRCNN(self.conf.IMAGE_SHAPE, pool_shape=[7,7],
#                          num_classes=self.conf.NUM_CLASSES,
#                          levels=[2,3,4,5], proposals=proposals,
#                          feature_maps=[fpn_graph['fpn_p2'], fpn_graph['fpn_p3'],
#                                        fpn_graph['fpn_p4'], fpn_graph['fpn_p5']],
#                          type='keras', DEBUG=False).get_mrcnn_graph()
#         # TODO: Set trainable weights for only non-trainable parameters
#         # TODO: DETECTION LAYER
#         # RPN has two losses 1) Classification loss and 2) Regularization
#
#         # TODO: Create RPN LOSS
#         # RPN has two losses 1) Classification loss and 2) Regularization
#         # rpn_loss = Loss.rpn_class_loss(rpn_target_class, rpn_class_logits)
#
#         # TODO: DETECTION
#
#         # TODO: Create MRCNN Loss  (Hmm how would we do it, when we havent compute the ground truth)
#
#         return fpn_graph, rpn_class_logits, rpn_class_probs, rpn_bbox, proposals, mrcnn_graph, xIN, anchors, \
#                rpn_target_class
#
#
#     def exec_sess(self, data_dict, image_ids):
#         # TODO: Inputs anchors and xIN
#         tf.reset_default_graph()
#
#         # BUILD THE GRAPH
#         fpn_graph, rpn_class_logits, rpn_class_probs, rpn_bbox, proposals, mrcnn_graph, xIN, anchors, rpn_target_class = self.build_train_graph()
#
#         # GET INPUT DATA
#         batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes, batch_image_metas, batch_rpn_target_class,\
#         batch_rpn_target_bbox, anchors_ = self.transform_images(data_dict, image_ids)
#
#         print ('batch_rpn_target_class ', batch_rpn_target_class)
#
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#
#             rpn_class_logits_, rpn_class_probs_, rpn_bbox_, proposals_, mrcnn_class_probs_, mrcnn_bbox_ = sess.run([rpn_class_logits, rpn_class_probs, rpn_bbox, proposals, mrcnn_graph['mrcnn_class_probs'], mrcnn_graph['mrcnn_bbox']],feed_dict={xIN: batch_images, anchors: anchors_})
#
#             print('Max and Min Proposals, ', np.amax(proposals_), np.amin(proposals_))
#             print('Num NaN present in Proposals ', np.sum(np.isnan(proposals_)))
#
#             print('(MRCNN) proposals (shape) ', proposals_.shape)
#
#
#             print(rpn_class_probs_.shape, rpn_bbox_.shape, mrcnn_class_probs_.shape, mrcnn_bbox_.shape)
#
#             print ('rpn_class_logits_.shape ', rpn_class_logits_.shape)
#             print ('')
#             print ('rpn_bbox_.shape ', rpn_bbox_.shape)
#             print('')
#             print ('mrcnn_class_probs_.shape ', mrcnn_class_probs_.shape)
#             print ('')
#             print('mrcnn_bbox_.shape' , mrcnn_bbox_.shape)
#             print ('batch_rpn_target_class ', batch_rpn_target_class.shape )
#
#             # target_class_ = sess.run(rpn_loss, feed_dict={rpn_target_class : batch_rpn_target_class})
#         #     print ('')
#         #     print('')
#         #     print(target_class_)
#         #



'''
Traceback (most recent call last):
  File "/Users/sam/All-Program/App/ObjectDetection/MaskRCNN/shapes.py", line 233, in <module>
    obj_trn.exec_sess(data_dict, image_ids)
  File "/Users/sam/All-Program/App/ObjectDetection/MaskRCNN/training.py", line 171, in exec_sess
    load_params.set_pretrained_weights(sess, self.pretrained_weights_path)
  File "/Users/sam/All-Program/App/ObjectDetection/MaskRCNN/building_blocks/load_params.py", line 117, in set_pretrained_weights
    raise ValueError('Mismatch is shape of pretrained weights and network defined weights')
ValueError: Mismatch is shape of pretrained weights and network defined weights
Variable:  <tf.Variable 'mrcnn_class_logits/kernel:0' shape=(1024, 4) dtype=float32_ref>
(1024, 81) (1024, 4)








Variable:  conv1/kernel:0
Shape:  (7, 7, 3, 64)
Variable:  conv1/bias:0
Shape:  (64,)
Variable:  bn_conv1/gamma:0
Shape:  (64,)
Variable:  bn_conv1/beta:0
Shape:  (64,)
Variable:  bn_conv1/moving_mean:0
Shape:  (64,)
Variable:  bn_conv1/moving_variance:0
Shape:  (64,)
Variable:  res2a_branch1/kernel:0
Shape:  (1, 1, 64, 256)
Variable:  res2a_branch1/bias:0
Shape:  (256,)
Variable:  bn2a_branch1/gamma:0
Shape:  (256,)
Variable:  bn2a_branch1/beta:0
Shape:  (256,)
Variable:  bn2a_branch1/moving_mean:0
Shape:  (256,)
Variable:  bn2a_branch1/moving_variance:0
Shape:  (256,)
Variable:  res2a_branch2a/kernel:0
Shape:  (1, 1, 64, 64)
Variable:  res2a_branch2a/bias:0
Shape:  (64,)
Variable:  bn2a_branch2a/gamma:0
Shape:  (64,)
Variable:  bn2a_branch2a/beta:0
Shape:  (64,)
Variable:  bn2a_branch2a/moving_mean:0
Shape:  (64,)
Variable:  bn2a_branch2a/moving_variance:0
Shape:  (64,)
Variable:  res2a_branch2b/kernel:0
Shape:  (3, 3, 64, 64)
Variable:  res2a_branch2b/bias:0
Shape:  (64,)
Variable:  bn2a_branch2b/gamma:0
Shape:  (64,)
Variable:  bn2a_branch2b/beta:0
Shape:  (64,)
Variable:  bn2a_branch2b/moving_mean:0
Shape:  (64,)
Variable:  bn2a_branch2b/moving_variance:0
Shape:  (64,)
Variable:  res2a_branch2c/kernel:0
Shape:  (1, 1, 64, 256)
Variable:  res2a_branch2c/bias:0
Shape:  (256,)
Variable:  bn2a_branch2c/gamma:0
Shape:  (256,)
Variable:  bn2a_branch2c/beta:0
Shape:  (256,)
Variable:  bn2a_branch2c/moving_mean:0
Shape:  (256,)
Variable:  bn2a_branch2c/moving_variance:0
Shape:  (256,)
Variable:  res2b_branch2a/kernel:0
Shape:  (1, 1, 256, 64)
Variable:  res2b_branch2a/bias:0
Shape:  (64,)
Variable:  bn2b_branch2a/gamma:0
Shape:  (64,)
Variable:  bn2b_branch2a/beta:0
Shape:  (64,)
Variable:  bn2b_branch2a/moving_mean:0
Shape:  (64,)
Variable:  bn2b_branch2a/moving_variance:0
Shape:  (64,)
Variable:  res2b_branch2b/kernel:0
Shape:  (3, 3, 64, 64)
Variable:  res2b_branch2b/bias:0
Shape:  (64,)
Variable:  bn2b_branch2b/gamma:0
Shape:  (64,)
Variable:  bn2b_branch2b/beta:0
Shape:  (64,)
Variable:  bn2b_branch2b/moving_mean:0
Shape:  (64,)
Variable:  bn2b_branch2b/moving_variance:0
Shape:  (64,)
Variable:  res2b_branch2c/kernel:0
Shape:  (1, 1, 64, 256)
Variable:  res2b_branch2c/bias:0
Shape:  (256,)
Variable:  bn2b_branch2c/gamma:0
Shape:  (256,)
Variable:  bn2b_branch2c/beta:0
Shape:  (256,)
Variable:  bn2b_branch2c/moving_mean:0
Shape:  (256,)
Variable:  bn2b_branch2c/moving_variance:0
Shape:  (256,)
Variable:  res2c_branch2a/kernel:0
Shape:  (1, 1, 256, 64)
Variable:  res2c_branch2a/bias:0
Shape:  (64,)
Variable:  bn2c_branch2a/gamma:0
Shape:  (64,)
Variable:  bn2c_branch2a/beta:0
Shape:  (64,)
Variable:  bn2c_branch2a/moving_mean:0
Shape:  (64,)
Variable:  bn2c_branch2a/moving_variance:0
Shape:  (64,)
Variable:  res2c_branch2b/kernel:0
Shape:  (3, 3, 64, 64)
Variable:  res2c_branch2b/bias:0
Shape:  (64,)
Variable:  bn2c_branch2b/gamma:0
Shape:  (64,)
Variable:  bn2c_branch2b/beta:0
Shape:  (64,)
Variable:  bn2c_branch2b/moving_mean:0
Shape:  (64,)
Variable:  bn2c_branch2b/moving_variance:0
Shape:  (64,)
Variable:  res2c_branch2c/kernel:0
Shape:  (1, 1, 64, 256)
Variable:  res2c_branch2c/bias:0
Shape:  (256,)
Variable:  bn2c_branch2c/gamma:0
Shape:  (256,)
Variable:  bn2c_branch2c/beta:0
Shape:  (256,)
Variable:  bn2c_branch2c/moving_mean:0
Shape:  (256,)
Variable:  bn2c_branch2c/moving_variance:0
Shape:  (256,)
Variable:  res3a_branch1/kernel:0
Shape:  (1, 1, 256, 512)
Variable:  res3a_branch1/bias:0
Shape:  (512,)
Variable:  bn3a_branch1/gamma:0
Shape:  (512,)
Variable:  bn3a_branch1/beta:0
Shape:  (512,)
Variable:  bn3a_branch1/moving_mean:0
Shape:  (512,)
Variable:  bn3a_branch1/moving_variance:0
Shape:  (512,)
Variable:  res3a_branch2a/kernel:0
Shape:  (1, 1, 256, 128)
Variable:  res3a_branch2a/bias:0
Shape:  (128,)
Variable:  bn3a_branch2a/gamma:0
Shape:  (128,)
Variable:  bn3a_branch2a/beta:0
Shape:  (128,)
Variable:  bn3a_branch2a/moving_mean:0
Shape:  (128,)
Variable:  bn3a_branch2a/moving_variance:0
Shape:  (128,)
Variable:  res3a_branch2b/kernel:0
Shape:  (3, 3, 128, 128)
Variable:  res3a_branch2b/bias:0
Shape:  (128,)
Variable:  bn3a_branch2b/gamma:0
Shape:  (128,)
Variable:  bn3a_branch2b/beta:0
Shape:  (128,)
Variable:  bn3a_branch2b/moving_mean:0
Shape:  (128,)
Variable:  bn3a_branch2b/moving_variance:0
Shape:  (128,)
Variable:  res3a_branch2c/kernel:0
Shape:  (1, 1, 128, 512)
Variable:  res3a_branch2c/bias:0
Shape:  (512,)
Variable:  bn3a_branch2c/gamma:0
Shape:  (512,)
Variable:  bn3a_branch2c/beta:0
Shape:  (512,)
Variable:  bn3a_branch2c/moving_mean:0
Shape:  (512,)
Variable:  bn3a_branch2c/moving_variance:0
Shape:  (512,)
Variable:  res3b_branch2a/kernel:0
Shape:  (1, 1, 512, 128)
Variable:  res3b_branch2a/bias:0
Shape:  (128,)
Variable:  bn3b_branch2a/gamma:0
Shape:  (128,)
Variable:  bn3b_branch2a/beta:0
Shape:  (128,)
Variable:  bn3b_branch2a/moving_mean:0
Shape:  (128,)
Variable:  bn3b_branch2a/moving_variance:0
Shape:  (128,)
Variable:  res3b_branch2b/kernel:0
Shape:  (3, 3, 128, 128)
Variable:  res3b_branch2b/bias:0
Shape:  (128,)
Variable:  bn3b_branch2b/gamma:0
Shape:  (128,)
Variable:  bn3b_branch2b/beta:0
Shape:  (128,)
Variable:  bn3b_branch2b/moving_mean:0
Shape:  (128,)
Variable:  bn3b_branch2b/moving_variance:0
Shape:  (128,)
Variable:  res3b_branch2c/kernel:0
Shape:  (1, 1, 128, 512)
Variable:  res3b_branch2c/bias:0
Shape:  (512,)
Variable:  bn3b_branch2c/gamma:0
Shape:  (512,)
Variable:  bn3b_branch2c/beta:0
Shape:  (512,)
Variable:  bn3b_branch2c/moving_mean:0
Shape:  (512,)
Variable:  bn3b_branch2c/moving_variance:0
Shape:  (512,)
Variable:  res3c_branch2a/kernel:0
Shape:  (1, 1, 512, 128)
Variable:  res3c_branch2a/bias:0
Shape:  (128,)
Variable:  bn3c_branch2a/gamma:0
Shape:  (128,)
Variable:  bn3c_branch2a/beta:0
Shape:  (128,)
Variable:  bn3c_branch2a/moving_mean:0
Shape:  (128,)
Variable:  bn3c_branch2a/moving_variance:0
Shape:  (128,)
Variable:  res3c_branch2b/kernel:0
Shape:  (3, 3, 128, 128)
Variable:  res3c_branch2b/bias:0
Shape:  (128,)
Variable:  bn3c_branch2b/gamma:0
Shape:  (128,)
Variable:  bn3c_branch2b/beta:0
Shape:  (128,)
Variable:  bn3c_branch2b/moving_mean:0
Shape:  (128,)
Variable:  bn3c_branch2b/moving_variance:0
Shape:  (128,)
Variable:  res3c_branch2c/kernel:0
Shape:  (1, 1, 128, 512)
Variable:  res3c_branch2c/bias:0
Shape:  (512,)
Variable:  bn3c_branch2c/gamma:0
Shape:  (512,)
Variable:  bn3c_branch2c/beta:0
Shape:  (512,)
Variable:  bn3c_branch2c/moving_mean:0
Shape:  (512,)
Variable:  bn3c_branch2c/moving_variance:0
Shape:  (512,)
Variable:  res3d_branch2a/kernel:0
Shape:  (1, 1, 512, 128)
Variable:  res3d_branch2a/bias:0
Shape:  (128,)
Variable:  bn3d_branch2a/gamma:0
Shape:  (128,)
Variable:  bn3d_branch2a/beta:0
Shape:  (128,)
Variable:  bn3d_branch2a/moving_mean:0
Shape:  (128,)
Variable:  bn3d_branch2a/moving_variance:0
Shape:  (128,)
Variable:  res3d_branch2b/kernel:0
Shape:  (3, 3, 128, 128)
Variable:  res3d_branch2b/bias:0
Shape:  (128,)
Variable:  bn3d_branch2b/gamma:0
Shape:  (128,)
Variable:  bn3d_branch2b/beta:0
Shape:  (128,)
Variable:  bn3d_branch2b/moving_mean:0
Shape:  (128,)
Variable:  bn3d_branch2b/moving_variance:0
Shape:  (128,)
Variable:  res3d_branch2c/kernel:0
Shape:  (1, 1, 128, 512)
Variable:  res3d_branch2c/bias:0
Shape:  (512,)
Variable:  bn3d_branch2c/gamma:0
Shape:  (512,)
Variable:  bn3d_branch2c/beta:0
Shape:  (512,)
Variable:  bn3d_branch2c/moving_mean:0
Shape:  (512,)
Variable:  bn3d_branch2c/moving_variance:0
Shape:  (512,)
Variable:  res4a_branch1/kernel:0
Shape:  (1, 1, 512, 1024)
Variable:  res4a_branch1/bias:0
Shape:  (1024,)
Variable:  bn4a_branch1/gamma:0
Shape:  (1024,)
Variable:  bn4a_branch1/beta:0
Shape:  (1024,)
Variable:  bn4a_branch1/moving_mean:0
Shape:  (1024,)
Variable:  bn4a_branch1/moving_variance:0
Shape:  (1024,)
Variable:  res4a_branch2a/kernel:0
Shape:  (1, 1, 512, 256)
Variable:  res4a_branch2a/bias:0
Shape:  (256,)
Variable:  bn4a_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4a_branch2a/beta:0
Shape:  (256,)
Variable:  bn4a_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4a_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4a_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4a_branch2b/bias:0
Shape:  (256,)
Variable:  bn4a_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4a_branch2b/beta:0
Shape:  (256,)
Variable:  bn4a_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4a_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4a_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4a_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4a_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4a_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4a_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4a_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4b_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4b_branch2a/bias:0
Shape:  (256,)
Variable:  bn4b_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4b_branch2a/beta:0
Shape:  (256,)
Variable:  bn4b_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4b_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4b_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4b_branch2b/bias:0
Shape:  (256,)
Variable:  bn4b_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4b_branch2b/beta:0
Shape:  (256,)
Variable:  bn4b_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4b_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4b_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4b_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4b_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4b_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4b_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4b_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4c_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4c_branch2a/bias:0
Shape:  (256,)
Variable:  bn4c_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4c_branch2a/beta:0
Shape:  (256,)
Variable:  bn4c_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4c_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4c_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4c_branch2b/bias:0
Shape:  (256,)
Variable:  bn4c_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4c_branch2b/beta:0
Shape:  (256,)
Variable:  bn4c_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4c_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4c_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4c_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4c_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4c_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4c_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4c_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4d_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4d_branch2a/bias:0
Shape:  (256,)
Variable:  bn4d_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4d_branch2a/beta:0
Shape:  (256,)
Variable:  bn4d_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4d_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4d_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4d_branch2b/bias:0
Shape:  (256,)
Variable:  bn4d_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4d_branch2b/beta:0
Shape:  (256,)
Variable:  bn4d_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4d_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4d_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4d_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4d_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4d_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4d_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4d_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4e_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4e_branch2a/bias:0
Shape:  (256,)
Variable:  bn4e_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4e_branch2a/beta:0
Shape:  (256,)
Variable:  bn4e_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4e_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4e_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4e_branch2b/bias:0
Shape:  (256,)
Variable:  bn4e_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4e_branch2b/beta:0
Shape:  (256,)
Variable:  bn4e_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4e_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4e_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4e_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4e_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4e_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4e_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4e_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4f_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4f_branch2a/bias:0
Shape:  (256,)
Variable:  bn4f_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4f_branch2a/beta:0
Shape:  (256,)
Variable:  bn4f_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4f_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4f_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4f_branch2b/bias:0
Shape:  (256,)
Variable:  bn4f_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4f_branch2b/beta:0
Shape:  (256,)
Variable:  bn4f_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4f_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4f_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4f_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4f_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4f_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4f_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4f_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4g_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4g_branch2a/bias:0
Shape:  (256,)
Variable:  bn4g_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4g_branch2a/beta:0
Shape:  (256,)
Variable:  bn4g_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4g_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4g_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4g_branch2b/bias:0
Shape:  (256,)
Variable:  bn4g_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4g_branch2b/beta:0
Shape:  (256,)
Variable:  bn4g_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4g_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4g_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4g_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4g_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4g_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4g_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4g_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4h_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4h_branch2a/bias:0
Shape:  (256,)
Variable:  bn4h_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4h_branch2a/beta:0
Shape:  (256,)
Variable:  bn4h_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4h_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4h_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4h_branch2b/bias:0
Shape:  (256,)
Variable:  bn4h_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4h_branch2b/beta:0
Shape:  (256,)
Variable:  bn4h_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4h_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4h_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4h_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4h_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4h_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4h_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4h_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4i_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4i_branch2a/bias:0
Shape:  (256,)
Variable:  bn4i_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4i_branch2a/beta:0
Shape:  (256,)
Variable:  bn4i_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4i_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4i_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4i_branch2b/bias:0
Shape:  (256,)
Variable:  bn4i_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4i_branch2b/beta:0
Shape:  (256,)
Variable:  bn4i_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4i_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4i_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4i_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4i_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4i_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4i_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4i_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4j_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4j_branch2a/bias:0
Shape:  (256,)
Variable:  bn4j_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4j_branch2a/beta:0
Shape:  (256,)
Variable:  bn4j_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4j_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4j_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4j_branch2b/bias:0
Shape:  (256,)
Variable:  bn4j_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4j_branch2b/beta:0
Shape:  (256,)
Variable:  bn4j_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4j_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4j_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4j_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4j_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4j_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4j_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4j_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4k_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4k_branch2a/bias:0
Shape:  (256,)
Variable:  bn4k_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4k_branch2a/beta:0
Shape:  (256,)
Variable:  bn4k_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4k_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4k_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4k_branch2b/bias:0
Shape:  (256,)
Variable:  bn4k_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4k_branch2b/beta:0
Shape:  (256,)
Variable:  bn4k_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4k_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4k_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4k_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4k_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4k_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4k_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4k_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4l_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4l_branch2a/bias:0
Shape:  (256,)
Variable:  bn4l_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4l_branch2a/beta:0
Shape:  (256,)
Variable:  bn4l_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4l_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4l_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4l_branch2b/bias:0
Shape:  (256,)
Variable:  bn4l_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4l_branch2b/beta:0
Shape:  (256,)
Variable:  bn4l_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4l_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4l_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4l_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4l_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4l_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4l_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4l_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4m_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4m_branch2a/bias:0
Shape:  (256,)
Variable:  bn4m_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4m_branch2a/beta:0
Shape:  (256,)
Variable:  bn4m_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4m_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4m_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4m_branch2b/bias:0
Shape:  (256,)
Variable:  bn4m_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4m_branch2b/beta:0
Shape:  (256,)
Variable:  bn4m_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4m_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4m_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4m_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4m_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4m_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4m_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4m_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4n_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4n_branch2a/bias:0
Shape:  (256,)
Variable:  bn4n_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4n_branch2a/beta:0
Shape:  (256,)
Variable:  bn4n_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4n_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4n_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4n_branch2b/bias:0
Shape:  (256,)
Variable:  bn4n_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4n_branch2b/beta:0
Shape:  (256,)
Variable:  bn4n_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4n_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4n_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4n_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4n_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4n_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4n_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4n_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4o_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4o_branch2a/bias:0
Shape:  (256,)
Variable:  bn4o_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4o_branch2a/beta:0
Shape:  (256,)
Variable:  bn4o_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4o_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4o_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4o_branch2b/bias:0
Shape:  (256,)
Variable:  bn4o_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4o_branch2b/beta:0
Shape:  (256,)
Variable:  bn4o_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4o_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4o_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4o_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4o_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4o_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4o_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4o_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4p_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4p_branch2a/bias:0
Shape:  (256,)
Variable:  bn4p_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4p_branch2a/beta:0
Shape:  (256,)
Variable:  bn4p_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4p_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4p_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4p_branch2b/bias:0
Shape:  (256,)
Variable:  bn4p_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4p_branch2b/beta:0
Shape:  (256,)
Variable:  bn4p_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4p_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4p_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4p_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4p_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4p_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4p_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4p_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4q_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4q_branch2a/bias:0
Shape:  (256,)
Variable:  bn4q_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4q_branch2a/beta:0
Shape:  (256,)
Variable:  bn4q_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4q_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4q_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4q_branch2b/bias:0
Shape:  (256,)
Variable:  bn4q_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4q_branch2b/beta:0
Shape:  (256,)
Variable:  bn4q_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4q_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4q_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4q_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4q_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4q_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4q_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4q_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4r_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4r_branch2a/bias:0
Shape:  (256,)
Variable:  bn4r_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4r_branch2a/beta:0
Shape:  (256,)
Variable:  bn4r_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4r_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4r_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4r_branch2b/bias:0
Shape:  (256,)
Variable:  bn4r_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4r_branch2b/beta:0
Shape:  (256,)
Variable:  bn4r_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4r_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4r_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4r_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4r_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4r_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4r_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4r_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4s_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4s_branch2a/bias:0
Shape:  (256,)
Variable:  bn4s_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4s_branch2a/beta:0
Shape:  (256,)
Variable:  bn4s_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4s_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4s_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4s_branch2b/bias:0
Shape:  (256,)
Variable:  bn4s_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4s_branch2b/beta:0
Shape:  (256,)
Variable:  bn4s_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4s_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4s_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4s_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4s_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4s_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4s_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4s_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4t_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4t_branch2a/bias:0
Shape:  (256,)
Variable:  bn4t_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4t_branch2a/beta:0
Shape:  (256,)
Variable:  bn4t_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4t_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4t_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4t_branch2b/bias:0
Shape:  (256,)
Variable:  bn4t_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4t_branch2b/beta:0
Shape:  (256,)
Variable:  bn4t_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4t_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4t_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4t_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4t_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4t_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4t_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4t_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4u_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4u_branch2a/bias:0
Shape:  (256,)
Variable:  bn4u_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4u_branch2a/beta:0
Shape:  (256,)
Variable:  bn4u_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4u_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4u_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4u_branch2b/bias:0
Shape:  (256,)
Variable:  bn4u_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4u_branch2b/beta:0
Shape:  (256,)
Variable:  bn4u_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4u_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4u_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4u_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4u_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4u_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4u_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4u_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4v_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4v_branch2a/bias:0
Shape:  (256,)
Variable:  bn4v_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4v_branch2a/beta:0
Shape:  (256,)
Variable:  bn4v_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4v_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4v_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4v_branch2b/bias:0
Shape:  (256,)
Variable:  bn4v_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4v_branch2b/beta:0
Shape:  (256,)
Variable:  bn4v_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4v_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4v_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4v_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4v_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4v_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4v_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4v_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res4w_branch2a/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  res4w_branch2a/bias:0
Shape:  (256,)
Variable:  bn4w_branch2a/gamma:0
Shape:  (256,)
Variable:  bn4w_branch2a/beta:0
Shape:  (256,)
Variable:  bn4w_branch2a/moving_mean:0
Shape:  (256,)
Variable:  bn4w_branch2a/moving_variance:0
Shape:  (256,)
Variable:  res4w_branch2b/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  res4w_branch2b/bias:0
Shape:  (256,)
Variable:  bn4w_branch2b/gamma:0
Shape:  (256,)
Variable:  bn4w_branch2b/beta:0
Shape:  (256,)
Variable:  bn4w_branch2b/moving_mean:0
Shape:  (256,)
Variable:  bn4w_branch2b/moving_variance:0
Shape:  (256,)
Variable:  res4w_branch2c/kernel:0
Shape:  (1, 1, 256, 1024)
Variable:  res4w_branch2c/bias:0
Shape:  (1024,)
Variable:  bn4w_branch2c/gamma:0
Shape:  (1024,)
Variable:  bn4w_branch2c/beta:0
Shape:  (1024,)
Variable:  bn4w_branch2c/moving_mean:0
Shape:  (1024,)
Variable:  bn4w_branch2c/moving_variance:0
Shape:  (1024,)
Variable:  res5a_branch1/kernel:0
Shape:  (1, 1, 1024, 2048)
Variable:  res5a_branch1/bias:0
Shape:  (2048,)
Variable:  bn5a_branch1/gamma:0
Shape:  (2048,)
Variable:  bn5a_branch1/beta:0
Shape:  (2048,)
Variable:  bn5a_branch1/moving_mean:0
Shape:  (2048,)
Variable:  bn5a_branch1/moving_variance:0
Shape:  (2048,)
Variable:  res5a_branch2a/kernel:0
Shape:  (1, 1, 1024, 512)
Variable:  res5a_branch2a/bias:0
Shape:  (512,)
Variable:  bn5a_branch2a/gamma:0
Shape:  (512,)
Variable:  bn5a_branch2a/beta:0
Shape:  (512,)
Variable:  bn5a_branch2a/moving_mean:0
Shape:  (512,)
Variable:  bn5a_branch2a/moving_variance:0
Shape:  (512,)
Variable:  res5a_branch2b/kernel:0
Shape:  (3, 3, 512, 512)
Variable:  res5a_branch2b/bias:0
Shape:  (512,)
Variable:  bn5a_branch2b/gamma:0
Shape:  (512,)
Variable:  bn5a_branch2b/beta:0
Shape:  (512,)
Variable:  bn5a_branch2b/moving_mean:0
Shape:  (512,)
Variable:  bn5a_branch2b/moving_variance:0
Shape:  (512,)
Variable:  res5a_branch2c/kernel:0
Shape:  (1, 1, 512, 2048)
Variable:  res5a_branch2c/bias:0
Shape:  (2048,)
Variable:  bn5a_branch2c/gamma:0
Shape:  (2048,)
Variable:  bn5a_branch2c/beta:0
Shape:  (2048,)
Variable:  bn5a_branch2c/moving_mean:0
Shape:  (2048,)
Variable:  bn5a_branch2c/moving_variance:0
Shape:  (2048,)
Variable:  res5b_branch2a/kernel:0
Shape:  (1, 1, 2048, 512)
Variable:  res5b_branch2a/bias:0
Shape:  (512,)
Variable:  bn5b_branch2a/gamma:0
Shape:  (512,)
Variable:  bn5b_branch2a/beta:0
Shape:  (512,)
Variable:  bn5b_branch2a/moving_mean:0
Shape:  (512,)
Variable:  bn5b_branch2a/moving_variance:0
Shape:  (512,)
Variable:  res5b_branch2b/kernel:0
Shape:  (3, 3, 512, 512)
Variable:  res5b_branch2b/bias:0
Shape:  (512,)
Variable:  bn5b_branch2b/gamma:0
Shape:  (512,)
Variable:  bn5b_branch2b/beta:0
Shape:  (512,)
Variable:  bn5b_branch2b/moving_mean:0
Shape:  (512,)
Variable:  bn5b_branch2b/moving_variance:0
Shape:  (512,)
Variable:  res5b_branch2c/kernel:0
Shape:  (1, 1, 512, 2048)
Variable:  res5b_branch2c/bias:0
Shape:  (2048,)
Variable:  bn5b_branch2c/gamma:0
Shape:  (2048,)
Variable:  bn5b_branch2c/beta:0
Shape:  (2048,)
Variable:  bn5b_branch2c/moving_mean:0
Shape:  (2048,)
Variable:  bn5b_branch2c/moving_variance:0
Shape:  (2048,)
Variable:  res5c_branch2a/kernel:0
Shape:  (1, 1, 2048, 512)
Variable:  res5c_branch2a/bias:0
Shape:  (512,)
Variable:  bn5c_branch2a/gamma:0
Shape:  (512,)
Variable:  bn5c_branch2a/beta:0
Shape:  (512,)
Variable:  bn5c_branch2a/moving_mean:0
Shape:  (512,)
Variable:  bn5c_branch2a/moving_variance:0
Shape:  (512,)
Variable:  res5c_branch2b/kernel:0
Shape:  (3, 3, 512, 512)
Variable:  res5c_branch2b/bias:0
Shape:  (512,)
Variable:  bn5c_branch2b/gamma:0
Shape:  (512,)
Variable:  bn5c_branch2b/beta:0
Shape:  (512,)
Variable:  bn5c_branch2b/moving_mean:0
Shape:  (512,)
Variable:  bn5c_branch2b/moving_variance:0
Shape:  (512,)
Variable:  res5c_branch2c/kernel:0
Shape:  (1, 1, 512, 2048)
Variable:  res5c_branch2c/bias:0
Shape:  (2048,)
Variable:  bn5c_branch2c/gamma:0
Shape:  (2048,)
Variable:  bn5c_branch2c/beta:0
Shape:  (2048,)
Variable:  bn5c_branch2c/moving_mean:0
Shape:  (2048,)
Variable:  bn5c_branch2c/moving_variance:0
Shape:  (2048,)
Variable:  fpn_c5p5/kernel:0
Shape:  (1, 1, 2048, 256)
Variable:  fpn_c5p5/bias:0
Shape:  (256,)
Variable:  fpn_c4p4/kernel:0
Shape:  (1, 1, 1024, 256)
Variable:  fpn_c4p4/bias:0
Shape:  (256,)
Variable:  fpn_c3p3/kernel:0
Shape:  (1, 1, 512, 256)
Variable:  fpn_c3p3/bias:0
Shape:  (256,)
Variable:  fpn_c2p2/kernel:0
Shape:  (1, 1, 256, 256)
Variable:  fpn_c2p2/bias:0
Shape:  (256,)
Variable:  fpn_p2/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  fpn_p2/bias:0
Shape:  (256,)
Variable:  fpn_p3/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  fpn_p3/bias:0
Shape:  (256,)
Variable:  fpn_p4/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  fpn_p4/bias:0
Shape:  (256,)
Variable:  fpn_p5/kernel:0
Shape:  (3, 3, 256, 256)
Variable:  fpn_p5/bias:0
Shape:  (256,)
Variable:  rpn_conv_shared/kernel:0
Shape:  (3, 3, 256, 512)
Variable:  rpn_conv_shared/bias:0
Shape:  (512,)
Variable:  rpn_class_raw/kernel:0
Shape:  (1, 1, 512, 6)
Variable:  rpn_class_raw/bias:0
Shape:  (6,)
Variable:  rpn_bbox_pred/kernel:0
Shape:  (1, 1, 512, 12)
Variable:  rpn_bbox_pred/bias:0
Shape:  (12,)
Variable:  mrcnn_class_conv1/kernel:0
Shape:  (7, 7, 256, 1024)
Variable:  mrcnn_class_conv1/bias:0
Shape:  (1024,)
Variable:  mrcnn_class_bn1/gamma:0
Shape:  (1024,)
Variable:  mrcnn_class_bn1/beta:0
Shape:  (1024,)
Variable:  mrcnn_class_bn1/moving_mean:0
Shape:  (1024,)
Variable:  mrcnn_class_bn1/moving_variance:0
Shape:  (1024,)
Variable:  mrcnn_class_conv2/kernel:0
Shape:  (1, 1, 1024, 1024)
Variable:  mrcnn_class_conv2/bias:0
Shape:  (1024,)
Variable:  mrcnn_class_bn2/gamma:0
Shape:  (1024,)
Variable:  mrcnn_class_bn2/beta:0
Shape:  (1024,)
Variable:  mrcnn_class_bn2/moving_mean:0
Shape:  (1024,)
Variable:  mrcnn_class_bn2/moving_variance:0
Shape:  (1024,)
Variable:  mrcnn_class_logits/kernel:0
Shape:  (1024, 4)
Variable:  mrcnn_class_logits/bias:0
Shape:  (4,)
Variable:  mrcnn_bbox_fc/kernel:0
Shape:  (1024, 16)
Variable:  mrcnn_bbox_fc/bias:0
Shape:  (16,)
'''
