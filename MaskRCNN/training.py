import os
import logging

import pickle
import numpy as np
import h5py
from scipy import ndimage
from scipy import misc
import tensorflow as tf
import keras.backend as K
import keras.layers as KL
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


class Train():
    def __init__(self, conf, batch_size, pretrained_weights_path):
        self.batch_size = batch_size
        self.pretrained_weights_path = pretrained_weights_path
        self.conf = conf
        print(self.conf)
    
    def build(self):
        pass
    
    def get_anchors(self, image_shape):
        logging.info('MaskRCNN get_anchors.....')
        """Returns anchor pyramid for the given image size."""
        
    
    def fetch_data(self, data_dict):
        self.batch_images = data_dict['batch_images']
        self.batch_gt_masks = data_dict['batch_gt_masks']
        self.batch_gt_class_ids = data_dict['batch_gt_class_ids']
        self.batch_gt_bboxes = data_dict['batch_gt_bboxes']
        self.batch_image_metas = data_dict['batch_image_metas']
        self.batch_rpn_target_class = data_dict['batch_rpn_target_class']
        self.batch_rpn_target_bbox = data_dict['batch_rpn_target_bbox']

        image_shape = self.batch_images.shape[1:]
        resnet_stage_shapes = utils.get_resnet_stage_shapes(self.conf, image_shape)


        self.anchors_normed = utils.gen_anchors(image_shape=image_shape,
                                    batch_size=self.batch_images.shape[0],
                                    scales=self.conf.RPN_ANCHOR_SCALES,
                                    ratios=self.conf.RPN_ANCHOR_RATIOS,
                                    feature_map_shapes=resnet_stage_shapes,
                                    feature_map_strides=self.conf.RESNET_STRIDES,
                                    anchor_strides=self.conf.RPN_ANCHOR_STRIDE)
            



    def get_detection_target_graph(self, proposals_, input_gt_class_ids, input_gt_boxes_norm):
        batch_rois = []
        batch_rois_gt_class_ids = []
        batch_rois_gt_class_boxes = []
        for i in range(0, self.batch_size):
            rois, roi_gt_class_ids, roi_gt_box_deltas = data_processor.BuildDetectionTargets(
                    self.conf, proposals_[i], input_gt_class_ids[i], input_gt_boxes_norm[i],
                    DEBUG=False).get_target_rois()
            
            batch_rois.append(rois)
            batch_rois_gt_class_ids.append(roi_gt_class_ids)
            batch_rois_gt_class_boxes.append(roi_gt_box_deltas)
            
            batch_rois = tf.stack(batch_rois, axis=0)
            batch_rois_gt_class_ids = tf.concat(batch_rois_gt_class_ids, axis=0)
            batch_rois_gt_class_boxes = tf.stack(batch_rois_gt_class_boxes, axis=0)
        
        return batch_rois, batch_rois_gt_class_ids, batch_rois_gt_class_boxes
    
    def get_network_inputs(self):
        
        ''' NOTE

        The gt_class_ids, gt_bboxes, are zero padded. This means that we may have actually only found 1
        gt_bbox and 1
        gt_class_ids, and rest 99 are just added to stack them.

        :return:
        '''
        
        self.xIN = tf.placeholder(dtype=tf.float32,
                                  shape=[None] + self.conf.IMAGE_SHAPE,
                                  name='input_image')
        
        # self.gt_masks = tf.placeholder(dtype=tf.float32,
        #                                shape=[None] + self.conf.IMAGE_SHAPE[:2] + [
        #                                    self.conf.MAX_GT_OBJECTS],
        #                                name='batch_gt_masks')
        # self.gt_class_ids = tf.placeholder(dtype=tf.int32,
        #                                    shape=[None, self.conf.MAX_GT_OBJECTS],
        #                                    name='gt_class_ids')
        
        # self.gt_bboxes = tf.placeholder(dtype=tf.float32,
        #                                 shape=[None, self.conf.MAX_GT_OBJECTS, 4],
        #                                 name='gt_bboxes')
        
        # Build Tensors for RPN Targets:
        self.rpn_target_class = tf.placeholder(dtype=tf.float32,
                                               shape=[None, None, 1],
                                               name='rpn_target_class')
        
        self.rpn_target_bbox = tf.placeholder(dtype=tf.float32,
                                              shape=[None, None, 4],
                                              name='rpn_target_bbox')
        
        self.anchors = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None, 4],
                                      name="input_anchors")
        
        # Build Tensors for Detection(MRCNN) target:
        self.input_gt_class_ids = tf.placeholder(dtype=tf.int32,
                                                 shape=[None, None],
                                                 name='input_gt_class_ids')
        
        self.input_gt_boxes_pxl = tf.placeholder(dtype=tf.float32,
                                              shape=[None, None, 4],
                                              name='input_gt_boxes')
        
        ##### Normalize the boxes to make proper comparison with Proposals
        self.input_gt_boxes_norm = [tf.expand_dims(utils.norm_boxes_tf(self.input_gt_boxes_pxl[i],
                                                        img_shape=self.conf.IMAGE_SHAPE[:2]), axis=0)
                                    for i in range(0, self.batch_size)]

        self.input_gt_boxes_norm = tf.concat(self.input_gt_boxes_norm, axis=0)

        # self.input_gt_boxes_norm = KL.Lambda(lambda x: utils.norm_boxes_tf(
        #         x, self.conf.IMAGE_SHAPE[:2]))(self.input_gt_boxes_pxl)

        
    
    def fpn_rpn_module(self):
        self.fpn_graph = FPN(self.conf, self.xIN, 'resnet101').get_fpn_graph()
    
        # CREATE THE RPN GRAPH
        rpn_pred_logits = []
        rpn_pred_probs = []
        rpn_pred_bbox = []
        for fmap in [self.fpn_graph['fpn_p2'],
                     self.fpn_graph['fpn_p3'],
                     self.fpn_graph['fpn_p4'],
                     self.fpn_graph['fpn_p5'],
                     self.fpn_graph['fpn_p6']]:
            rpn_obj = RPN(self.conf, depth=256, feature_map=fmap)
            rpn_pred_logits.append(rpn_obj.get_rpn_class_logits())
            rpn_pred_probs.append(rpn_obj.get_rpn_class_probs())
            rpn_pred_bbox.append(rpn_obj.get_rpn_bbox())
    
        self.rpn_pred_logits = tf.concat(rpn_pred_logits, axis=1)
        self.rpn_pred_logits = tf.concat(rpn_pred_logits, axis=1)
        self.rpn_pred_probs = tf.concat(rpn_pred_probs, axis=1)
        self.rpn_pred_bbox = tf.concat(rpn_pred_bbox, axis=1)
        
    def proposal_module(self):
        # CREATE THE PROPOSAL GRAPH
        self.proposals = Proposals(
                self.conf,
                self.batch_size,
                self.rpn_pred_probs,
                self.rpn_pred_bbox,
                self.anchors, training=True, DEBUG=True
        ).get_proposals()
    
    def detection_target(self):
        self.rois, self.mrcnn_target_class_ids, self.mrcnn_target_box = self.get_detection_target_graph(
                self.proposals,
                self.input_gt_class_ids,
                self.input_gt_boxes_norm)
        
    def mrcnn_module(self):
        self.mrcnn_graph = MaskRCNN(self.conf.IMAGE_SHAPE,
                               pool_shape=[7, 7],
                               num_classes=self.conf.NUM_CLASSES,
                               levels=[2, 3, 4, 5],
                               proposals=self.rois,
                               feature_maps=[self.fpn_graph['fpn_p2'], self.fpn_graph['fpn_p3'],
                                             self.fpn_graph['fpn_p4'], self.fpn_graph['fpn_p5']],
                               type='keras', DEBUG=False).get_mrcnn_graph()
        
    def build_train_graph(self, batch_active_class_ids):
    
        self.get_network_inputs()
        
        # Module FPN RPN
        self.fpn_rpn_module()
        
        # Module Proposal
        self.proposal_module()
        
        # Fetching Detection Targets
        self.detection_target()

        # Module MRCNN
        self.mrcnn_module()

        # TODO: Create RPN LOSS
        # RPN has two losses 1) Classification loss and 2) Regularization
        self.rpn_class_loss = Loss.rpn_class_loss(self.rpn_target_class, self.rpn_pred_logits)
        self.rpn_loss_extra_var, self.rpn_box_loss = Loss.rpn_box_loss(
                self.rpn_target_bbox,
                self.rpn_pred_bbox,
                self.rpn_target_class,
                batch_size=self.batch_size)

        self.mrcnn_loss_extra_var, self.mrcnn_class_loss = Loss.mrcnn_class_loss(
                mrcnn_target_class_ids=self.mrcnn_target_class_ids,
                mrcnn_pred_logits=self.mrcnn_graph['mrcnn_class_logits'],
                batch_active_class_ids=batch_active_class_ids
        )

        # TODO: Create MRCNN Loss  (Hmm how would we do it, when we haven't compute the ground truth)
        self.mrcnn_box_loss = Loss.mrcnn_box_loss(
                mrcnn_target_box=self.mrcnn_target_box,
                mrcnn_pred_box=self.mrcnn_graph['mrcnn_bbox'],
                mrcnn_target_class_ids=self.mrcnn_target_class_ids,
                batch_size=self.batch_size)
            
    def exec_sess(self, data_dict, image_ids):
        # TODO: Inputs anchors and xIN
        tf.reset_default_graph()
        
        # GET INPUT DATA
        self.fetch_data(data_dict)
        
        batch_active_class_ids = self.batch_image_metas[:, -4:]  # 1 corresponds to the active level
        
        # BUILD THE GRAPH
        self.build_train_graph(batch_active_class_ids)
        
        print('batch_active_class_ids ', batch_active_class_ids.shape, batch_active_class_ids)
        print('batch_rpn_target_class ', self.batch_rpn_target_class.shape)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # PRINT ALL THE TRAINING VARIABLES
            load_params.print_trainable_variable_names(sess)
            
            # GET PRETRAINED WEIGHTS
            if self.pretrained_weights_path:
                load_params.set_pretrained_weights(sess, self.pretrained_weights_path,
                                                   train_nets='heads')

            # a = sess.run(self.input_gt_boxes_norm, feed_dict = {
            #          self.xIN: self.transformed_images,
            #          self.anchors: self.anchors_normed,
            #          self.input_gt_class_ids: self.batch_gt_class_ids,
            #          self.input_gt_boxes_pxl: self.batch_gt_bboxes,
            #          self.rpn_target_class: self.batch_rpn_target_class,
            #          self.rpn_target_bbox: self.batch_rpn_target_bbox
            #          })
            # print ('batch_gt_bboxes ', self.batch_gt_bboxes.shape, self.batch_gt_bboxes)
            # print ('sdsdsd ', a.shape, a)
            
            
            outputs = [self.rpn_pred_logits,
                       self.rpn_pred_probs,
                       self.rpn_pred_bbox,
                       # self.proposals,
                       # self.mrcnn_target_class_ids,
                       # self.mrcnn_target_box,
                       # self.mrcnn_graph['mrcnn_class_logits'],
                       # self.mrcnn_graph['mrcnn_class_probs'],
                       # self.mrcnn_graph['mrcnn_bbox'],
                       # self.rpn_loss_extra_var,
                       # self.rpn_class_loss,
                       # self.rpn_box_loss,
                       # self.mrcnn_class_loss,
                       # self.mrcnn_box_loss
                       ]



            feed_dict = {self.xIN: self.batch_images,
                         self.anchors: self.anchors_normed,
                         self.input_gt_class_ids: self.batch_gt_class_ids,
                         self.input_gt_boxes_pxl: self.batch_gt_bboxes,
                         self.rpn_target_class: self.batch_rpn_target_class,
                         self.rpn_target_bbox: self.batch_rpn_target_bbox
                         }

            outputs_ = sess.run(outputs, feed_dict=feed_dict)

            self.outputs_ = outputs_

            # print('Max and Min Proposals, ', np.amax(outputs_[3]), np.amin(outputs_[3]))
            # print('Num NaN present in Proposals ', np.sum(np.isnan(outputs_[3])))

            print('(RPN) rpn_pred_logits: ', outputs_[0].shape)
            print('(RPN) rpn_pred_probs: ', outputs_[1].shape)
            print('(RPN) rpn_pred_bbox: ', outputs_[2].shape)
            # print('(PROPOSAL) proposals ', outputs_[3].shape)
            # print('(MRCNN) mrcnn_target_class_ids', outputs_[4].shape)
            # print('(MRCNN) mrcnn_target_box (shape) ', outputs_[5].shape)
            # print('(MRCNN) mrcnn_class_logits', outputs_[6].shape)
            # print('(MRCNN) mrcnn_class_probs (shape) ', outputs_[7].shape)
            # print('(MRCNN) mrcnn_bbox (shape) ', outputs_[8].shape)
            # print('(LOSS) rpn_loss_extra_var (shape) ', outputs_[9].shape)
            # print('(LOSS) rpn_class_loss (shape) ', outputs_[10])
            # print('(LOSS) rpn_box_loss (shape) ', outputs_[11])
            # print('(LOSS) mrcnn_class_loss (shape) ', outputs_[12])
            # print('(LOSS) mrcnn_box_loss (shape) ', outputs_[13])
            #
            #
            # print('batch_rpn_target_class: ', self.batch_rpn_target_class)



            # print(self.batch_image_metas)

            # print(outputs_[9])