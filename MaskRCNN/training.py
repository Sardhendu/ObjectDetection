
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

        transformed_images, image_metas, image_windows, anchors = data_processor.process_images(self.conf,
                                                                                                batch_images,
                                                                                                image_ids)

        print(transformed_images.shape, image_metas.shape, image_windows.shape,
              anchors.shape)

        return (batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes, batch_image_metas,
                batch_rpn_target_class,
                batch_rpn_target_bbox, anchors)

    def get_detection_target_graph(self, proposals_, input_gt_class_ids, input_gt_bboxes):
        batch_rois = []
        batch_rois_gt_class_ids = []
        batch_rois_gt_class_boxes = []
        for i in range(0, self.batch_size):
            rois, roi_gt_class_ids, roi_gt_box_deltas = data_processor.BuildDetectionTargets(
                    self.conf, proposals_[i], input_gt_class_ids[i], input_gt_bboxes[i],
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

        self.gt_masks = tf.placeholder(dtype=tf.float32,
                                       shape=[None] + self.conf.IMAGE_SHAPE[:2] + [
                                           self.conf.MAX_GT_OBJECTS],
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
        # (xIN, gt_masks, gt_class_ids, gt_bboxes, rpn_target_class, rpn_target_bbox, anchors,
        # input_gt_class_ids, input_gt_bboxes) = \
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
        (rois, mrcnn_target_class_ids, mrcnn_target_box) = self.get_detection_target_graph(
                proposals,
                self.input_gt_class_ids,
                self.input_gt_bboxes)

        print('74823749234 ', mrcnn_target_class_ids.dtype)
        print('09090909090 ', mrcnn_target_box.dtype)

        # MRCNN GRAPH
        mrcnn_graph = MaskRCNN(self.conf.IMAGE_SHAPE,
                               pool_shape=[7, 7],
                               num_classes=self.conf.NUM_CLASSES,
                               levels=[2, 3, 4, 5],
                               proposals=rois,
                               feature_maps=[fpn_graph['fpn_p2'], fpn_graph['fpn_p3'],
                                             fpn_graph['fpn_p4'], fpn_graph['fpn_p5']],
                               type='keras', DEBUG=False).get_mrcnn_graph()

        # TODO: Create RPN LOSS
        # RPN has two losses 1) Classification loss and 2) Regularization
        rpn_class_loss = Loss.rpn_class_loss(self.rpn_target_class, rpn_pred_logits)
        rpn_target_bbox_nopad, rpn_pred_box_pos = Loss.rpn_box_loss(
                self.rpn_target_bbox, rpn_pred_bbox, self.rpn_target_class,
                batch_size=self.batch_size)

        mrcnn_class_loss = Loss.mrcnn_class_loss(
                mrcnn_target_class_ids=mrcnn_target_class_ids,
                mrcnn_pred_logits=mrcnn_graph['mrcnn_class_logits'],
                batch_active_class_ids=batch_active_class_ids
        )

        # TODO: Create MRCNN Loss  (Hmm how would we do it, when we havent compute the ground truth)

        return (fpn_graph, rpn_pred_logits, rpn_pred_probs, rpn_pred_bbox, proposals,
                mrcnn_graph, rpn_class_loss, rpn_target_bbox_nopad, rpn_pred_box_pos,
                mrcnn_class_loss, mrcnn_target_class_ids, mrcnn_target_box)

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
        (fpn_graph, rpn_pred_logits, rpn_pred_probs, rpn_pred_bbox,
         proposals, mrcnn_graph,
         rpn_class_loss, rpn_target_bbox_nopad, rpn_pred_box_pos,
         mrcnn_class_loss, mrcnn_target_class_ids, mrcnn_target_box_pos) = self.build_train_graph(
            batch_active_class_ids)

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
    
            (rpn_pred_logits_, rpn_pred_probs_, rpn_pred_bbox_, proposals_, mrcnn_target_class_ids_,
             mrcnn_target_box_,
             mrcnn_class_logits_, mrcnn_class_probs_, mrcnn_bbox_,
             rpn_class_loss_, rpn_box_loss_, mrcnn_class_loss_) = sess.run([
                rpn_pred_logits,
                rpn_pred_probs,
                rpn_pred_bbox,
                proposals,
                mrcnn_target_class_ids,
                mrcnn_target_box_pos,
                mrcnn_graph['mrcnn_class_logits'],
                mrcnn_graph['mrcnn_class_probs'],
                mrcnn_graph['mrcnn_bbox'],
                rpn_class_loss,
                rpn_pred_box_pos,
                mrcnn_class_loss],
                    feed_dict={self.xIN: batch_images,
                               self.anchors: anchors_,
                               self.input_gt_class_ids: batch_gt_class_ids,
                               self.input_gt_bboxes: batch_gt_bboxes,
                               self.rpn_target_class: batch_rpn_target_class,
                               self.rpn_target_bbox: batch_rpn_target_bbox
                               })
    
            print('Max and Min Proposals, ', np.amax(proposals_), np.amin(proposals_))
            print('Num NaN present in Proposals ', np.sum(np.isnan(proposals_)))
    
            print('(RPN) pred_logits: ', rpn_pred_logits.shape)
            print('(RPN) pred_prob: ', rpn_pred_probs.shape)
            print('(RPN) pred_box: ', rpn_pred_bbox.shape)
            print('proposals (shape) ', proposals_.shape)
            print('(MRCNN) target_class_ids', mrcnn_target_class_ids_.shape)
            print('(MRCNN) target_box (shape) ', mrcnn_target_box_.shape)
            print('(MRCNN) pred_class_ids', mrcnn_class_probs_.shape)
            print('(MRCNN) pred_box (shape) ', mrcnn_bbox_.shape)

    
            print('rpn_class_loss_: ', rpn_class_loss_)
    
            # print ('RUNNNING MRCNN CLASS LOSS')
            #
            # mrcnn_class_loss_ = sess.run(mrcnn_class_loss, feed_dict={self.xIN: batch_images,
            #                    self.anchors: anchors_,
            #                    self.input_gt_class_ids: batch_gt_class_ids,
            #                    self.input_gt_bboxes: batch_gt_bboxes})
            #
            print('mrcnn_class_loss_: ', mrcnn_class_loss_)
    
            print('')
            print(batch_image_metas)
    
            ### MRCNN box loss
            Loss.mrcnn_box_loss(mrcnn_target_box_, mrcnn_bbox_, mrcnn_target_class_ids_)
    
    
    
            # Loss.mrcnn_class_loss(mrcnn_target_class_ids=mrcnn_target_class_ids_,
            #                       mrcnn_pred_logits=mrcnn_class_logits_,
            #                       batch_active_class_ids=batch_active_class_ids,
            #                       sess=sess)
    
            ######### ROUGH ###########################
            # from MaskRCNN.building_blocks import data_processor
            # proposal__ = np.array([[[1, 10, 1, 10], [23, 54, 155, 177], [10,10,167,170],
            #                         [0, 0, 0, 0]],
            #                        [[3, 2, 2, 2], [54, 22, 144, 171], [0,0,0,0],[0, 0, 0, 0]]])
            #
            # # proposals_ = tf.placeholder(shape=(2, 3, 4), dtype=tf.float32, name='proposals')
            # proposals_ = tf.placeholder(shape=(2, 4, 4), dtype=tf.float32, name='proposals')
            #
            # batch_rois = []
            # batch_rois_gt_class_ids = []
            # batch_rois_gt_class_boxes = []
            # for i in range(0, 2):
            #     rois, roi_gt_class_ids, roi_gt_box_deltas = data_processor.BuildDetectionTargets(
            # self.conf,
            #             proposals_[i], batch_gt_bboxes[i], batch_gt_class_ids[i],
            # DEBUG=False).get_target_rois()
            #
            #     batch_rois.append(rois)
            #     batch_rois_gt_class_ids.append(roi_gt_class_ids)
            #     batch_rois_gt_class_boxes.append(roi_gt_box_deltas)
            #
            # batch_rois = tf.stack(batch_rois, axis=0)
            # batch_rois_gt_class_ids = tf.concat(batch_rois_gt_class_ids, axis=0)
            # batch_rois_gt_class_boxes = tf.stack(batch_rois_gt_class_boxes, axis=0)
            #
            # print (batch_rois.get_shape().as_list())
            # print(batch_rois_gt_class_ids.get_shape().as_list())
            # print(batch_rois_gt_class_boxes.get_shape().as_list())
            #
            # # batch_rois = tf.stack([rois], )
            # # break
            # # batch_proposals.append(prop)
            #
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     #
            #     a,b, c = sess.run([batch_rois, batch_rois_gt_class_ids, batch_rois_gt_class_boxes],
            #              feed_dict={proposals_:proposal__})
            #
            #
            #     # a, b, c= sess.run([rois, roi_gt_box_deltas],
            #     #                       feed_dict={proposals_: proposal__})
            #
            #     print(a.shape)
            #     print (b.shape)
            #     print(c.shape)
            #
            # ##########################################
    
    
    
    
    
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





























































            
            
            