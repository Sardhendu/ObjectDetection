import pickle
import numpy as np
import h5py
from scipy import ndimage
from scipy import misc
import tensorflow as tf

import logging

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
        print(self.conf)
    
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
        
        transformed_images, image_metas, image_windows, anchors = data_processor.process_images(self.conf, batch_images,
                                                                                                image_ids)
        
        print(transformed_images.shape, image_metas.shape, image_windows.shape,
              anchors.shape)
        
        return (
            batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes, batch_image_metas,
            batch_rpn_target_class,
            batch_rpn_target_bbox, anchors)
    
    def get_detection_target_graph(self, proposals_, input_gt_class_ids, input_gt_bboxes):
        batch_rois = []
        batch_rois_gt_class_ids = []
        batch_rois_gt_class_boxes = []
        for i in range(0, self.batch_size):
            dict_ = data_processor.BuildDetectionTargets(
                    self.conf, proposals_[i], input_gt_class_ids[i], input_gt_bboxes[i],
                    DEBUG=True).debug_outputs()
        
        return dict_
    
    def get_network_inputs(self):
        ''' NOTE

        The gt_class_ids, gt_bboxes, are zero padded. This means that we may have actually only found 1 gt_bbox and 1
        gt_class_ids, and rest 99 are just added to stack them.

        :return:
        '''
        self.xIN = tf.placeholder(dtype=tf.float32,
                                  shape=[None] + self.conf.IMAGE_SHAPE,
                                  name='input_image')
        
        self.gt_masks = tf.placeholder(dtype=tf.float32,
                                       shape=[None] + self.conf.IMAGE_SHAPE[:2] + [self.conf.MAX_GT_OBJECTS],
                                       name='batch_gt_masks')
        self.gt_class_ids = tf.placeholder(dtype=tf.int32,
                                           shape=[None, self.conf.MAX_GT_OBJECTS],
                                           name='gt_class_ids')
        
        self.gt_bboxes = tf.placeholder(dtype=tf.float32,
                                        shape=[None, self.conf.MAX_GT_OBJECTS, 4],
                                        name='gt_bboxes')
        
        self.rpn_target_class = tf.placeholder(dtype=tf.float32,
                                               shape=[None, None, 1],
                                               name='rpn_target_class')
        
        self.rpn_target_bbox = tf.placeholder(dtype=tf.float32,
                                              shape=[None, None, 4],
                                              name='rpn_target_bbox')
        
        self.anchors = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None, 4],
                                      name="input_anchors")
        
        # CREATE BATCH DATA: Needed to create the graph for Detection Targets
        self.input_gt_class_ids = tf.placeholder(dtype=tf.int32,
                                                 shape=[self.batch_size,
                                                        self.conf.DETECTION_POST_NMS_INSTANCES],
                                                 name='input_gt_class_ids')
        
        self.input_gt_bboxes = tf.placeholder(dtype=tf.float32,
                                              shape=[self.batch_size,
                                                     self.conf.DETECTION_POST_NMS_INSTANCES,
                                                     4],
                                              name='input_gt_boxes')
        
        # return (xIN, gt_masks, gt_class_ids, gt_bboxes, rpn_target_class,
        #         rpn_target_bbox, anchors, input_gt_class_ids, input_gt_bboxes)
    
    def build_train_graph(self, batch_active_class_ids):
        
        # GET GRAPH INPUTS
        # (xIN, gt_masks, gt_class_ids, gt_bboxes, rpn_target_class, rpn_target_bbox, anchors, input_gt_class_ids,
        # input_gt_bboxes) = \
        #
        self.get_network_inputs()
        
        # CREATE THE FPN GRAPH
        fpn_graph = FPN(self.conf, self.xIN, 'resnet101').get_fpn_graph()
        
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
                              rpn_pred_probs, rpn_pred_bbox,
                              self.anchors, training=True, DEBUG=True).get_proposals()
        
        # BUILD MRCNN/DETECTION TARGET GRAPH
        dict_ = self.get_detection_target_graph(
                proposals,
                self.input_gt_class_ids,
                self.input_gt_bboxes)
        
        # MRCNN GRAPH
        mrcnn_graph = MaskRCNN(self.conf.IMAGE_SHAPE,
                               pool_shape=[7, 7],
                               num_classes=self.conf.NUM_CLASSES,
                               levels=[2, 3, 4, 5],
                               proposals=dict_['l'],
                               feature_maps=[fpn_graph['fpn_p2'], fpn_graph['fpn_p3'],
                                             fpn_graph['fpn_p4'], fpn_graph['fpn_p5']],
                               type='keras', DEBUG=False).get_mrcnn_graph()
        #
        # TODO: Create RPN LOSS
        # # RPN has two losses 1) Classification loss and 2) Regularization
        # rpn_class_loss = Loss.rpn_class_loss(self.rpn_target_class, rpn_pred_logits)
        # rpn_target_bbox_nopad, rpn_pred_box_pos = Loss.rpn_box_loss(
        #         self.rpn_target_bbox, rpn_pred_bbox, self.rpn_target_class,
        #         batch_size=self.batch_size)
        #
        # mrcnn_class_loss = Loss.mrcnn_class_loss(
        #         mrcnn_target_class_ids=dict_['j'],
        #         mrcnn_pred_logits=mrcnn_graph['mrcnn_class_logits'],
        #         batch_active_class_ids=batch_active_class_ids
        # )
        #
        # # TODO: Create MRCNN Loss  (Hmm how would we do it, when we havent compute the ground truth)
        #
        # return (fpn_graph, rpn_pred_logits, rpn_pred_probs, rpn_pred_bbox, proposals,
        #         mrcnn_graph, rpn_class_loss, rpn_target_bbox_nopad, rpn_pred_box_pos,
        #         mrcnn_class_loss, dict_['j'], dict_['k'], dict)
        
        return dict_, mrcnn_graph
        
        # return rpn_pred_logits, rpn_pred_probs, rpn_pred_bbox, proposals
    
    def exec_sess(self, data_dict, image_ids):
        # TODO: Inputs anchors and xIN
        tf.reset_default_graph()
        
        # GET INPUT DATA
        (batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes,
         batch_image_metas, batch_rpn_target_class, batch_rpn_target_bbox,
         anchors_) = self.transform_images(data_dict, image_ids)
        
        batch_active_class_ids = batch_image_metas[:, -4:]  # 1 corresponds to the active level
        
        # BUILD THE GRAPH
        dict_, mrcnn_bbox = self.build_train_graph(batch_active_class_ids)

        
        print('batch_active_class_ids ', batch_active_class_ids.shape, batch_active_class_ids)
        print('batch_rpn_target_class ', batch_rpn_target_class.shape)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # PRINT ALL THE TRAINING VARIABLES
            load_params.print_trainable_variable_names(sess)
            
            # GET PRETRAINED WEIGHTS
            if self.pretrained_weights_path:
                load_params.set_pretrained_weights(sess, self.pretrained_weights_path,
                                                   train_nets='heads')
            
            # print('opopopoopopopopp ', batch_gt_class_ids.shape)
            # print('asdadadasdasdasdasdas ', batch_gt_class_ids.dtype, batch_images.dtype)
            
            (prop, gt_boxes, gt_class_ids, iou, pos_rois, neg_rois, pos_iou,
             roi_gt_box_assignment, roi_gt_class_ids_before, roi_gt_class_ids_final,
             roi_gt_box_deltas,
             rois ) = sess.run([
                dict_['a'],
                dict_['b'],
                dict_['c'],
                dict_['d'],
                dict_['e'],
                dict_['f'],
                dict_['g'],
                dict_['h'],
                dict_['i'],
                dict_['j'],
                dict_['k'],
                dict_['l']],
                    feed_dict={self.xIN: batch_images,
                               self.anchors: anchors_,
                               self.input_gt_class_ids: batch_gt_class_ids,
                               self.input_gt_bboxes: batch_gt_bboxes,
                               self.rpn_target_class: batch_rpn_target_class,
                               self.rpn_target_bbox: batch_rpn_target_bbox
                               })
            
            print('Proposals \n ', prop)
            print('')
            print('gt_boxes \n ', gt_boxes)
            print('')
            print('gt_class_ids \n ', gt_class_ids)
            print('')
            print('iou \n ', iou)
            print('')
            print('pos_rois \n ', pos_rois)
            print('')
            print('neg_rois \n ', neg_rois)
            print('')
            print('pos_iou \n ', pos_iou)
            print('')
            print('roi_gt_box_assignment \n ', roi_gt_box_assignment)
            print('')
            print('roi_gt_class_ids_before \n ', roi_gt_class_ids_before)
            print('')
            print('roi_gt_class_ids_final \n ', roi_gt_class_ids_final)
            print('')
            print('roi_gt_box_deltas \n ', roi_gt_box_deltas)
            print('')
            print('rois \n ', rois)
            
            
            # print('Max and Min Proposals, ', np.amax(proposals_), np.amin(proposals_))
            # print('Num NaN present in Proposals ', np.sum(np.isnan(proposals_)))
            #
            # print('(RPN) pred_logits: ', rpn_pred_logits.shape)
            # print('(RPN) pred_prob: ', rpn_pred_probs.shape)
            # print('(RPN) pred_box: ', rpn_pred_bbox.shape)
            # print('proposals (shape) ', proposals_.shape)
            # print('(MRCNN) target_class_ids', mrcnn_target_class_ids_.shape)
            # print('(MRCNN) target_box (shape) ', mrcnn_target_box_.shape)
            # print('(MRCNN) pred_class_ids', mrcnn_class_probs_.shape)
            # print('(MRCNN) pred_box (shape) ', mrcnn_bbox_.shape)
            
