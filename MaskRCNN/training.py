
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
    def __init__(self, conf, batch_size):
        self.batch_size = batch_size
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
        
        return \
        (batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes, batch_image_metas, batch_rpn_target_class,
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
        rpn_class_logits = []
        rpn_class_probs = []
        rpn_bbox = []
        for fmap in [fpn_graph['fpn_p2'], fpn_graph['fpn_p3'], fpn_graph['fpn_p4'], fpn_graph['fpn_p5'],
                     fpn_graph['fpn_p6']]:
            rpn_obj = RPN(self.conf, depth=256, feature_map=fmap)  #
            rpn_class_logits.append(rpn_obj.get_rpn_class_logits())
            rpn_class_probs.append(rpn_obj.get_rpn_class_probs())
            rpn_bbox.append(rpn_obj.get_rpn_bbox())
        
        rpn_class_logits = tf.concat(rpn_class_logits, axis=1)
        rpn_class_probs = tf.concat(rpn_class_probs, axis=1)
        rpn_bbox = tf.concat(rpn_bbox, axis=1)
        
        # CREATE THE PROPOSAL GRAPH
        proposals = Proposals(self.conf, self.batch_size, rpn_class_probs, rpn_bbox, anchors,
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
        rpn_loss = Loss.rpn_class_loss(rpn_target_class, rpn_class_logits)
        
        # TODO: DETECTION
        
        # TODO: Create MRCNN Loss  (Hmm how would we do it, when we havent compute the ground truth)
        
        return fpn_graph, rpn_class_logits, rpn_class_probs, rpn_bbox, proposals, mrcnn_graph, xIN, anchors, \
               rpn_target_class, rpn_loss
    
    def exec_sess(self, data_dict, image_ids):
        # TODO: Inputs anchors and xIN
        tf.reset_default_graph()
        
        # BUILD THE GRAPH
        fpn_graph, rpn_class_logits, rpn_class_probs, rpn_bbox, proposals, mrcnn_graph, xIN, anchors, \
        rpn_target_class, rpn_loss = self.build_train_graph()
        
        # GET INPUT DATA
        batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes, batch_image_metas, batch_rpn_target_class, \
        batch_rpn_target_bbox, anchors_ = self.transform_images(data_dict, image_ids)
        
        print('batch_rpn_target_class ', batch_rpn_target_class)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            rpn_class_logits_, rpn_class_probs_, rpn_bbox_, proposals_, mrcnn_class_probs_, mrcnn_bbox_ = sess.run(
                    [rpn_class_logits, rpn_class_probs, rpn_bbox, proposals, mrcnn_graph['mrcnn_class_probs'],
                     mrcnn_graph['mrcnn_bbox']],
                    feed_dict={xIN: batch_images, anchors: anchors_})
            
            print('Max and Min Proposals, ', np.amax(proposals_), np.amin(proposals_))
            print('Num NaN present in Proposals ', np.sum(np.isnan(proposals_)))
            
            print('(MRCNN) proposals (shape) ', proposals_.shape)
            
            # print(rpn_class_probs_.shape, rpn_bbox_.shape, mrcnn_class_probs_.shape, mrcnn_bbox_.shape)
            
            print('rpn_class_logits_.shape ', rpn_class_logits_.shape)
            print('')
            print ('rpn_bbox_.shape ', rpn_bbox_.shape)
            print('')
            print ('mrcnn_class_probs_.shape ', mrcnn_class_probs_.shape)
            print ('')
            print('mrcnn_bbox_.shape ', mrcnn_bbox_.shape)
            
            loss = sess.run(rpn_loss, feed_dict={rpn_target_class: batch_rpn_target_class,
                                                 rpn_class_logits:rpn_class_logits_})

            print(loss)
            print ('')
            print(batch_rpn_target_bbox)
            print('')
            
            
            for i in range(0,10):
                print(batch_rpn_target_bbox[:,i,:])
            
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
#
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