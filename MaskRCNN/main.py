import os
import logging

import pickle
import numpy as np
import h5py
from scipy import ndimage
from scipy import misc
import tensorflow as tf

from MaskRCNN.config import config as conf
from MaskRCNN.building_blocks import preprocess
from MaskRCNN.building_blocks import load_params
from MaskRCNN.building_blocks.fpn import FPN
from MaskRCNN.building_blocks.rpn import RPN
from MaskRCNN.building_blocks.proposals_tf import Proposals
from MaskRCNN.building_blocks.maskrcnn import MaskRCNN
from MaskRCNN.building_blocks.detection import DetectionLayer, unmold_detection
from MaskRCNN.building_blocks import utils

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def get_trainable_variable_name(sess):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Variable name: Shape: ", k, v.shape)


# https://github.com/CharlesShang/FastMaskRCNN



class Inference():
    def __init__(self, pretrained_weights_path, run, save, save_dir, DEBUG):
        self.DEBUG = DEBUG
        self.run = run
        self.save = save
        self.save_dir = save_dir
        self.pretrained_weights_path = pretrained_weights_path
        self.proposals = []
        self.feature_maps = []
        
        self.build()
    
    def build(self):
        if self.run == 'fpn_rcn_proposals':
            transformed_images, anchors = self.transform_image()
            print('(ANCHORS): ', anchors.shape)
            self.run_fpn_rcn_proposals(transformed_images, anchors)
        elif self.run == 'mrcnn_detection':
            self.run_mrcnn_detection()
    
    def transform_image(self):
        img_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/images/3627527276_6fe8cd9bfe_z.jpg'
        image = ndimage.imread(img_path, mode='RGB')
        self.original_image_shape = image.shape
        image_id = os.path.basename(img_path).split('.')[0]
        
        ## Process Images:
        list_of_images = [image]
        list_of_image_ids = [image_id]
        transformed_images, self.image_metas, self.image_windows, anchors = preprocess.process_images(
                list_of_images, list_of_image_ids
        )
        self.image_shape = transformed_images.shape[1:]
        self.batch_size = transformed_images.shape[0]
        print('(INPUT IMAGE) Shape of input image batch: ', self.image_shape)
        print('(IMAGE META) Image Metas: ', self.image_metas)
        return transformed_images, anchors
    
    def run_fpn_rcn_proposals(self, transformed_images, anchors):
        xIN = tf.placeholder(dtype=tf.float32,
                             shape=[None] + conf.IMAGE_SHAPE,
                             name='input_image')
        
        # CRATE THE FPN GRAPH
        fpn_graph = FPN(xIN, 'resnet101').get_fpn_graph()  # Basically the Resnet architecture.
        
        # CREATE THE RPN GRAPH
        rpn_graph = RPN(depth=256).get_rpn_graph()
        
        # CREATE THE PROPOSAL GRAPH
        proposal_graph = Proposals(conf, batch_size=self.batch_size).get_proposal_graph()
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            
            if self.DEBUG:
                # PRINT ALL THE TRAINABLE VARIALES
                get_trainable_variable_name(sess)
                load_params.set_pretrained_weights(sess, self.pretrained_weights_path)
            else:
                # Get input Image:
                # Note setting the weight can take 1-2 min due to the deep network
                load_params.set_pretrained_weights(sess, self.pretrained_weights_path)
                
                # RUN FPN GRAPH
                print('RUNNING FPN ..............')
                feed_dict = {xIN: transformed_images}  # np.random.random((batch_size, 1024, 1024, 3))}
                p2, p3, p4, p5, p6 = sess.run([fpn_graph['fpn_p2'], fpn_graph['fpn_p3'],
                                               fpn_graph['fpn_p4'], fpn_graph['fpn_p5'],
                                               fpn_graph['fpn_p6']], feed_dict=feed_dict)
                print('(FPN) P2=%s, P3=%s, P4=%s, P5=%s, P6=%s' % (
                    str(p2.shape), str(p3.shape), str(p4.shape), str(p5.shape), str(p6.shape)))
                
                # RUN RPN GRAPH
                print('RUNNING RPN ..............')
                rpn_class_probs = []
                rpn_bboxes = []
                for fpn_p in [p2, p3, p4, p5, p6]:
                    _, rpn_prob, rpn_bbox = sess.run([rpn_graph['rpn_class_logits'], rpn_graph['rpn_class_probs'],
                                                      rpn_graph['rpn_bbox']], feed_dict={rpn_graph['xrpn']: fpn_p})
                    rpn_class_probs.append(rpn_prob)
                    rpn_bboxes.append(rpn_bbox)
                    print('(RPN): rpn_class_score=%s, rpn_bbox=%s ' % (str(rpn_prob.shape), str(rpn_bbox.shape)))
                    del rpn_prob
                    del rpn_bbox
                
                self.feature_maps = [p2, p3, p4, p5]
                del p2
                del p3
                del p4
                del p5
                
                # Concatenate with the second dimension
                rpn_class_probs = np.concatenate(rpn_class_probs, axis=1)
                rpn_bboxes = np.concatenate(rpn_bboxes, axis=1)
                print('(RPN) Total(stacked): rpn_class_score=%s, rpn_bbox=%s ' % (str(rpn_class_probs.shape),
                                                                                  str(rpn_bboxes.shape)))
                
                # RUN PROPOSAL GRAPH AND THE MASK-RCNN GRAPH AND DETECTION TOGETHER
                # Get Anchors
                self.proposals = sess.run(proposal_graph['proposals'],
                                          feed_dict={proposal_graph['rpn_class_probs']: rpn_class_probs,
                                                     proposal_graph['rpn_bbox']: rpn_bboxes,
                                                     proposal_graph['input_anchors']: anchors})
                print('(PROPOSALS): ', self.proposals.shape)
        
        tf.reset_default_graph()
        
        if self.save:
            self.save_feature_maps_and_proposals()
            
            
    def run_mrcnn_detection(self):
        feature_maps, proposals, image_metas = self.get_feature_maps_and_proposals()
        print ('Max and Min Proposals, ', np.amax(proposals), np.amin(proposals))
        print ('Num NaN present in Proposals ', np.sum(np.isnan(proposals)))

        print('(MRCNN) feature_map (len) ', len(feature_maps))
        print('(MRCNN) proposals (shape) ', proposals.shape)


        # JUST FOR 1 IMAGE
        self.batch_size = len(image_metas)
        self.original_image_shape = np.array(image_metas[:,1:4], dtype='int32')
        self.image_shape = np.array(image_metas[:,4:7], dtype='int32')
        self.image_window = np.array(image_metas[:,7:11], dtype='int32')
        
        print (self.batch_size, self.original_image_shape, self.image_shape, self.image_window)
        
        # self.image_metas = np.array([['3627527276_6fe8cd9bfe_z', '476', '640', '3', '1024', '1024', '3', '131', '0', '893', '1155', '1.6', '0']])

        # proposalsIN = tf.placeholder(dtype=tf.float32, shape=proposals.shape, name='proposals')
        mrcnn_graph = MaskRCNN(image_shape=self.image_shape[0], pool_shape=[7, 7], num_classes=81,
                               levels=[2, 3, 4, 5], proposals=proposals, feature_maps=feature_maps,
                               type='keras', DEBUG=False).get_mrcnn_graph()

        detections = DetectionLayer(
                conf,
                image_shape=self.image_shape[0], # All images in a batch should be of same shape
                num_batches=self.batch_size,
                window=self.image_window,
                proposals=proposals, mrcnn_class_probs=mrcnn_graph['mrcnn_class_probs'],
                mrcnn_bbox=mrcnn_graph['mrcnn_bbox'], DEBUG=False).get_detections()


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if self.DEBUG:
                # PRINT ALL THE TRAINABLE VARIABLE
                get_trainable_variable_name(sess)
                load_params.set_pretrained_weights(sess, self.pretrained_weights_path)
            else:
                # Get input Image:
                # Note setting the weight can take 1-2 min due to the deep network
                load_params.set_pretrained_weights(sess, self.pretrained_weights_path)

                self.mrcnn_class_probs, self.mrcnn_bbox, self.detections_ = sess.run(
                        [mrcnn_graph['mrcnn_class_probs'],
                         mrcnn_graph['mrcnn_bbox'], detections])
                print('(MASK RCNN) mrcnn_class_probs (shape)', self.mrcnn_class_probs.shape)
                print('(MASK RCNN) mrcnn_bbox (shape)', self.mrcnn_bbox.shape)
                print('(DETECTION) detections (shape)', self.detections_.shape)

                self.detection_boxes = []
                for i in range(0, len(self.detections_)):
                    detections_unmold, class_ids, scores = unmold_detection(
                            original_image_shape=self.original_image_shape[i],
                            image_shape=self.image_shape[i],
                            detections=self.detections_[i],
                            image_window=self.image_window[i])
                    print (detections_unmold)
                    self.detection_boxes.append(detections_unmold)
        
        if self.save:
            self.save_mrcnn_prob_bbox_and_detections()

    def save_feature_maps_and_proposals(self):
    
        with open(os.path.join(self.save_dir, 'feature_maps_n_proposals.pickle'), "wb") as f:
            fullData = {
                'feature_maps': self.feature_maps,
                'proposals': self.proposals,
                'image_metas': self.image_metas
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)

    def get_feature_maps_and_proposals(self):
        if len(self.feature_maps) > 0 and len(self.proposals) > 0:
            return self.feature_maps, self.proposals, self.image_metas
        else:
            with open(os.path.join(self.save_dir, 'feature_maps_n_proposals.pickle'), "rb") as f:
                data = pickle.load(f)
                feature_maps = data['feature_maps']
                proposals = data['proposals']
                image_metas = data['image_metas']
            return feature_maps, proposals, image_metas

    def save_mrcnn_prob_bbox_and_detections(self):
        with open(os.path.join(self.save_dir, 'mrcnn_prob_bbox_detection.pickle'), "wb") as f:
            fullData = {
                'mrcnn_class_probs': self.mrcnn_class_probs,
                'mrcnn_bbox': self.mrcnn_bbox,
                'detections': self.detections_,
                'detection_unmold': self.detection_boxes
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)

    def get_mrcnn_prob_bbox_and_detections(self):
        with open(os.path.join(self.save_dir, 'mrcnn_prob_bbox_detection.pickle'), "rb") as f:
            data = pickle.load(f)
            mrcnn_class_probs = data['mrcnn_class_probs']
            mrcnn_bbox = data['mrcnn_bbox']
            detections = data['detections']
            detection_unmold = data['detection_unmold']
            
        return mrcnn_class_probs, mrcnn_bbox, detections, detection_unmold
        
        
        
pretrained_weights_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
save_dir = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/'

##### PRINT PRETRAINED WEIGHTS
# from MaskRCNN.building_blocks.load_params import print_pretrained_weights
# print_pretrained_weights(weights_path=pretrained_weights_path, search_key='mrcnn')



####### RUN FPN AND RPN
# obj_inference = Inference(pretrained_weights_path, run='fpn_rcn_proposals', save=True, save_dir=save_dir, DEBUG=False)
# feature_maps, proposals, image_metas = obj_inference.get_feature_maps_and_proposals()
# print ('len(feature_maps), proposal.shape ',len(feature_maps), proposals.shape)
# print('image_metas ', image_metas)


####### RUN MRCNN AND DETECTION
# obj_inference = Inference(pretrained_weights_path, run='mrcnn_detection', save=True, save_dir=save_dir, DEBUG=False)
# mrcnn_class_probs, mrcnn_bbox, detections, detection_unmold = obj_inference.get_mrcnn_prob_bbox_and_detections()
# print (detection_unmold)




# mrcnn_bbox_fc
# bias:0
# kernel:0
# mrcnn_class_bn1
# beta:0
# gamma:0
# moving_mean:0
# moving_variance:0
# mrcnn_class_bn2
# beta:0
# gamma:0
# moving_mean:0
# moving_variance:0
# mrcnn_class_conv1
# bias:0
# kernel:0
# mrcnn_class_conv2
# bias:0
# kernel:0
# mrcnn_class_logits
# bias:0
# kernel:0
