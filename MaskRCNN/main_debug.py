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
        if self.run == 'fpn_rpn':
            transformed_images, _ = self.transform_image()
            self.run_fpn_rpn(transformed_images)
        elif self.run == 'proposals':
            _, anchors = self.transform_image()
            self.run_proposals(anchors)
        elif self.run == 'mrcnn':
            self.run_mrcnn()
        elif self.run == 'detections':
            self.run_detection()
        else:
            raise ValueError('Provide a valid value for argument "run"')
    
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
    
    def run_fpn_rpn(self, transformed_images):
        xIN = tf.placeholder(dtype=tf.float32,
                             shape=[None] + conf.IMAGE_SHAPE,
                             name='input_image')
        
        # CRATE THE FPN GRAPH
        fpn_graph = FPN(xIN, 'resnet101').get_fpn_graph()  # Basically the Resnet architecture.
        
        # CREATE THE RPN GRAPH
        rpn_graph = RPN(depth=256).get_rpn_graph()
        
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
                # Concatenate with the second dimension
                self.rpn_class_probs = np.concatenate(rpn_class_probs, axis=1)
                self.rpn_bboxes = np.concatenate(rpn_bboxes, axis=1)
                print('(RPN) Total(stacked): rpn_class_score=%s, rpn_bbox=%s ' % (str(self.rpn_class_probs.shape),
                                                                                  str(self.rpn_bboxes.shape)))
                
        if self.save:
            self.save_feature_map()
            self.save_rpn_probs_bbox()
            self.save_image_metas()
            
    def run_proposals(self, anchors):
        rpn_class_probs, rpn_bboxes = self.get_rpn_probs_bbox()
        
        # CREATE THE PROPOSAL GRAPH
        if self.DEBUG:
            from MaskRCNN.building_blocks.proposals_tf import debug
            debug(rpn_class_probs=rpn_class_probs, rpn_bbox=rpn_bboxes, input_anchors=anchors)
        else:
            proposal_graph = Proposals(conf, batch_size=self.batch_size).get_proposal_graph()
    
            init = tf.global_variables_initializer()
    
            with tf.Session() as sess:
                sess.run(init)
                load_params.set_pretrained_weights(sess, self.pretrained_weights_path)
                self.proposals = sess.run(proposal_graph['proposals'],
                                          feed_dict={proposal_graph['rpn_class_probs']: rpn_class_probs,
                                                     proposal_graph['rpn_bbox']: rpn_bboxes,
                                                     proposal_graph['input_anchors']: anchors})
                print('(PROPOSALS): ', self.proposals.shape)
                
            if self.save:
                self.save_proposals()
    
    def run_mrcnn(self):
        image_metas = self.get_image_metas()
        feature_maps = self.get_feature_maps()
        proposals = self.get_proposals()

        image_shape = np.array(image_metas[:, 4:7], dtype='int32')
        print('Max and Min Proposals, ', np.amax(proposals), np.amin(proposals))
        print('Num NaN present in Proposals ', np.sum(np.isnan(proposals)))

        print('(MRCNN) feature_map (len) ', len(feature_maps))
        print('(MRCNN) proposals (shape) ', proposals.shape)
        
        
        if self.DEBUG:
            from MaskRCNN.building_blocks.maskrcnn import debug
            debug(feature_maps=feature_maps, proposals=proposals, image_metas=image_metas)
        else:
            mrcnn_graph = MaskRCNN(image_shape=image_shape[0], pool_shape=[7, 7], num_classes=81,
                                   levels=[2, 3, 4, 5], proposals=proposals, feature_maps=feature_maps,
                                   type='keras', DEBUG=False).get_mrcnn_graph()
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                load_params.set_pretrained_weights(sess, self.pretrained_weights_path)
                self.mrcnn_class_probs, self.mrcnn_bboxes = sess.run([mrcnn_graph['mrcnn_class_probs'],
                                                                    mrcnn_graph['mrcnn_bbox']])
                print('(MASK RCNN) mrcnn_class_probs (shape)', self.mrcnn_class_probs.shape)
                print('(MASK RCNN) mrcnn_bbox (shape)', self.mrcnn_bboxes.shape)
                
            if self.save:
                self.save_mrcnn_probs_bbox()
        
    def run_detection(self):
        image_metas = self.get_image_metas()
        proposals = self.get_proposals()
        mrcnn_class_probs, mrcnn_bboxes = self.get_mrcnn_probs_bbox()
    
        # JUST FOR 1 IMAGE
        batch_size = len(image_metas)
        original_image_shape = np.array(image_metas[:, 1:4], dtype='int32')
        image_shape = np.array(image_metas[:, 4:7], dtype='int32')
        image_window = np.array(image_metas[:, 7:11], dtype='int32')
        
        print(batch_size, original_image_shape, image_shape, image_window)
        


        if self.DEBUG:
            from MaskRCNN.building_blocks.detection import debug
            debug(proposals=proposals, mrcnn_class_probs=mrcnn_class_probs, mrcnn_bbox=mrcnn_bboxes,
                  image_window=image_window, image_shape=image_shape[0])
        else:
            detections = DetectionLayer(
                                    conf,
                                    image_shape=image_shape[0],  # All images in a batch should be of same shape
                                    num_batches=batch_size,
                                    window=image_window,
                                    proposals=proposals, mrcnn_class_probs=mrcnn_class_probs,
                                    mrcnn_bbox=mrcnn_bboxes, DEBUG=False).get_detections()
            
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
    
                # Get input Image:
                # Note setting the weight can take 1-2 min due to the deep network
                load_params.set_pretrained_weights(sess, self.pretrained_weights_path)
                
                self.detections_normed = sess.run(detections)
                print('(DETECTION) detections (shape)', self.detections_normed.shape)
        
            if self.save:
                self.save_detections()
                
    def save_image_metas(self):
        with open(os.path.join(self.save_dir, 'image_metas.pickle'), "wb") as f:
            fullData = {
                'image_metas': self.image_metas,
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)

    def save_feature_map(self):
        with open(os.path.join(self.save_dir, 'feature_maps.pickle'), "wb") as f:
            fullData = {
                'feature_maps': self.feature_maps,
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)
        
    def save_rpn_probs_bbox(self):
        with open(os.path.join(self.save_dir, 'rpn_probs_bboxes.pickle'), "wb") as f:
            fullData = {
                'rpn_class_probs': self.rpn_class_probs,
                'rpn_bboxes': self.rpn_bboxes
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)
    
    def save_proposals(self):
        with open(os.path.join(self.save_dir, 'proposals.pickle'), "wb") as f:
            fullData = {
                'proposals': self.proposals,
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)
        
    def save_mrcnn_probs_bbox(self):
        with open(os.path.join(self.save_dir, 'mrcnn_probs_bboxes.pickle'), "wb") as f:
            fullData = {
                'mrcnn_class_probs': self.mrcnn_class_probs,
                'mrcnn_bboxes': self.mrcnn_bboxes
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)

    def save_detections(self):
        with open(os.path.join(self.save_dir, 'detections.pickle'), "wb") as f:
            fullData = {
                'detections_normed': self.detections_normed
            }
            pickle.dump(fullData, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_image_metas(save_dir):
        with open(os.path.join(save_dir, 'image_metas.pickle'), "rb") as f:
            data = pickle.load(f)
            image_metas = data['image_metas']
        return image_metas

    @staticmethod
    def get_feature_maps(save_dir):
        with open(os.path.join(save_dir, 'feature_maps.pickle'), "rb") as f:
            data = pickle.load(f)
            feature_maps = data['feature_maps']
        return feature_maps

    @staticmethod
    def get_rpn_probs_bbox(save_dir):
        with open(os.path.join(save_dir, 'rpn_probs_bboxes.pickle'), "rb") as f:
            data = pickle.load(f)
            rpn_class_probs = data['rpn_class_probs']
            rpn_bboxes = data['rpn_bboxes']
        return rpn_class_probs, rpn_bboxes

    @staticmethod
    def get_proposals(save_dir):
        with open(os.path.join(save_dir, 'proposals.pickle'), "rb") as f:
            data = pickle.load(f)
            proposals = data['proposals']
        return proposals

    @staticmethod
    def get_mrcnn_probs_bbox(save_dir):
        with open(os.path.join(save_dir, 'mrcnn_probs_bboxes.pickle'), "rb") as f:
            data = pickle.load(f)
            mrcnn_class_probs = data['mrcnn_class_probs']
            mrcnn_bboxes = data['mrcnn_bboxes']
        return mrcnn_class_probs, mrcnn_bboxes

    @staticmethod
    def get_detections(save_dir):
        with open(os.path.join(save_dir, 'detections.pickle'), "rb") as f:
            data = pickle.load(f)
            detections_normed = data['detections_normed']
        return detections_normed

    @staticmethod
    def get_detection_dnormed(image_metas, detections_normed):
        print ('detections_normed.shape ', detections_normed.shape)
        original_image_shape = np.array(image_metas[:, 1:4], dtype='int32')
        image_shape = np.array(image_metas[:, 4:7], dtype='int32')
        image_window = np.array(image_metas[:, 7:11], dtype='int32')
    
        detections_dnormed = []
        class_ids = []
        scores = []
        for i in range(0, len(detections_normed)):
            detection_dnormed, class_id, score = unmold_detection(
                    original_image_shape=original_image_shape[i],
                    image_shape=image_shape[i],
                    detections=detections_normed[i],
                    image_window=image_window[i])
            detections_dnormed.append(detection_dnormed)
            class_ids.append(class_id)
            scores.append(score)
    
        return detections_dnormed, class_ids, scores


pretrained_weights_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/mask_rcnn_coco.h5'
save_dir = '/Users/sam/All-Program/App-DataSet/ObjectDetection/MaskRCNN/debug_outputs/'




######## RUN THE FPN RPN
# obj_inference = Inference(pretrained_weights_path, run='fpn_rpn', save=True, save_dir=save_dir,
# DEBUG=False)
# feature_maps = Inference.get_feature_maps(save_dir)
# rpn_class_probs, rpn_bboxes = Inference.get_rpn_probs_bbox(save_dir)
# print ('len(feature_maps) ',len(feature_maps))
# print ('rpn_class_probs.shape  ',rpn_class_probs.shape)
# print ('rpn_bboxes.shape  ',rpn_bboxes.shape)


######## RUN THE PROPOSALS MODULE
# obj_inference = Inference(pretrained_weights_path, run='proposals', save=True, save_dir=save_dir,
# DEBUG=True)
# proposals = Inference.get_proposals(save_dir)
# print ('proposals ',proposals.shape)


####### RUN THE MRCNN MODULE
# obj_inference = Inference(pretrained_weights_path, run='mrcnn', save=True, save_dir=save_dir,
# DEBUG=True)
# mrcnn_class_probs, mrcnn_bboxes = Inference.get_mrcnn_probs_bbox(save_dir)
# print ('mrcnn_class_probs ',mrcnn_class_probs.shape)
# print ('mrcnn_bboxes ',mrcnn_bboxes.shape)


####### RUN THE DETECTION MODULE
# obj_inference = Inference(pretrained_weights_path, run='detections', save=True, save_dir=save_dir,
# DEBUG=True)
# detections_normed = Inference.get_detections(save_dir)
# print ('detections_normed ',detections_normed.shape)



####### DNORM DETECTION AT PIXEL LEVEL
# image_metas = Inference.get_image_metas(save_dir)
# proposal_normed = Inference.get_proposals(save_dir)
# print (proposal_normed.shape)
# proposal_normed = np.expand_dims(np.column_stack((proposal_normed[0], np.array(np.ones((1000,2)), dtype='int32'))), axis=0)
# detections_dnormed, class_ids, scores = Inference.get_detection_dnormed(image_metas, proposal_normed)
# print ('detections_dnormed ',detections_dnormed[0])
# print ('class_ids ',class_ids[0])
# print ('scores ',scores[0])