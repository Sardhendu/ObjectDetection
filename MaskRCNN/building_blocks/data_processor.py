
'''
The RCNN module takes input as a certain format. For instance the

'''

from skimage.transform import resize
import numpy as np
import math
import tensorflow as tf
from MaskRCNN.building_blocks import utils


# Keep metadata of images for post processing
def compose_image_meta(image_id, original_image_shape, image_shape,
                       img_window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    img_window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(img_window) +                # size=4 (y1, x1, y2, x2) in image coordinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def process_images(conf, list_of_images, list_of_image_ids):
    ''' Prepare Test Data
    
    :param list_of_images:
    :param list_of_image_ids:
    
    :return: Three matrices
        1. resized_images = [num_images, h, w, 3]
        2. image_metas = details about image such as
            [image_id, orig_img_h, orig_img_w, orig_img_channel,
             transformed_img_h, transformed_img_w, transformed_img_channel,
             img_window_hTop, img_window_hBottom, img_window_wLeft, img_window_wRight,
             scale_factor, active_class_ids]
        3. image_windows = [img_window_hTop, img_window_hBottom, img_window_wLeft, img_window_wRight,]
        4. anchors =  Initial anchors
    '''
    
    # Reshape, Normalize images
    transformed_images = []
    image_metas = []
    image_windows = []
    for img, img_id in zip(list_of_images, list_of_image_ids):
        transformed_image, image_window, scale, padding = utils.resize_image(
                img, conf.IMAGE_MIN_DIM, conf.IMAGE_MAX_DIM, conf.IMAGE_MIN_SCALE, conf.IMAGE_RESIZE_MODE
        )

        transformed_image = utils.normalize_image(transformed_image, conf.MEAN_PIXEL)

        # COmpose a metadata of images for debugging and
        image_meta = compose_image_meta(
                img_id, img.shape, transformed_image.shape, image_window, scale,
                np.zeros([conf.NUM_CLASSES], dtype=np.int32))

        transformed_images.append(transformed_image)
        image_metas.append(image_meta)
        image_windows.append(image_window)

    transformed_images = np.stack(transformed_images)
    image_metas = np.stack(image_metas)
    image_windows = np.stack(image_windows)
    print ('(process_images) Image Resized Nnormalized: (shape) ', transformed_images.shape)
    print ('(process_images) Image meta: (shape) ', image_metas.shape)
    print ('(process_images) Image Window: (shape) ', image_windows.shape)
    
    
    # Generate Anchors
    resnet_stage_shapes = utils.get_resnet_stage_shapes(
            conf, image_shape=transformed_images.shape[1:]
    )
    anchors = utils.gen_anchors(image_shape=transformed_images.shape[1:],
                                batch_size=transformed_images.shape[0],
                                scales=conf.RPN_ANCHOR_SCALES,
                                ratios=conf.RPN_ANCHOR_RATIOS,
                                feature_map_shapes=resnet_stage_shapes,
                                feature_map_strides=conf.RESNET_STRIDES,
                                anchor_strides=conf.RPN_ANCHOR_STRIDE)

    print('(process_images) Anchors: (shape) ', anchors.shape)


    return transformed_images, image_metas, image_windows, anchors



class BuildDetectionTargets():
    def __init__(self, conf, proposals, gt_class_ids, gt_bboxes, DEBUG=False):
        self.train_rois_per_image = conf.MRCNN_TRAIN_ROIS_PER_IMAGE
        self.box_stddev = conf.BBOX_STD_DEV
        
        if DEBUG:
            self.DEBUG = True
        else:
            self.DEBUG = False
            
        self.build_detection_target(proposals, gt_class_ids, gt_bboxes)
        # self.detection_targets_graph(proposals, gt_bboxes, gt_class_ids)
        
        
    def box_refinement_tf(self, box, gt_box):
        """Compute refinement needed to transform box to gt_box.
        box and gt_box are [N, (y1, x1, y2, x2)]
        
        Theory:
            A foreground (+ve) proposals or anchor or roi might not be centered perfectly over
            the object. So the RPN estimates a delta (% change in x, y, width, height) to
            refine the anchor box w.r.t gt_boxes to fit the object better.
        """
        box = tf.cast(box, tf.float32)
        gt_box = tf.cast(gt_box, tf.float32)
    
        height = box[:, 2] - box[:, 0]
        width = box[:, 3] - box[:, 1]
        center_y = box[:, 0] + 0.5 * height
        center_x = box[:, 1] + 0.5 * width
    
        gt_height = gt_box[:, 2] - gt_box[:, 0]
        gt_width = gt_box[:, 3] - gt_box[:, 1]
        gt_center_y = gt_box[:, 0] + 0.5 * gt_height
        gt_center_x = gt_box[:, 1] + 0.5 * gt_width
    
        dy = (gt_center_y - center_y) / height
        dx = (gt_center_x - center_x) / width
        dh = tf.log(gt_height / height)
        dw = tf.log(gt_width / width)
    
        result = tf.stack([dy, dx, dh, dw], axis=1)
        return result
    
        
    def get_iou_tf(self, proposal_per_img, gt_boxes_per_img):
        '''Get Intersection over union with tensorflow
        
        :param proposal_per_img:
        :param gt_boxes_per_img:
        :return:                    nxm matrix
                                    where n = num active proposals per image
                                          m = num active gt_boxes_per_image
        '''
        # Repeat the proposals "gt_boxes" time
        proposal_ = tf.reshape(      # Reshapes to original shape but repeating values
                             tf.tile(       # repeats the values horizontally
                                tf.expand_dims(proposal_per_img, 1),
                                [1, 1, tf.shape(gt_boxes_per_img)[0]]),
                            [-1,4])
        
        # repeat gt_boxes vertically to match proposal_per_img
        gt_boxes_ = tf.tile(gt_boxes_per_img, [tf.shape(proposal_per_img)[0], 1])
        
        # Get the coordinates
        p_y1, p_x1, p_y2, p_x2 = tf.split(proposal_, 4, axis=1)
        g_y1, g_x1, g_y2, g_x2 = tf.split(gt_boxes_, 4, axis=1)
        
        # Get the Area
        p_area = tf.multiply(p_y2 - p_y1, p_x2 - p_x1)
        g_area = tf.multiply(g_y2 - g_y1, g_x2 - g_x1)
        
        # Get Intersection
        i_y1 = tf.maximum(p_y1, g_y1)
        i_x1 = tf.maximum(p_x1, g_x1)
        i_y2 = tf.minimum(p_y2, g_y2)
        i_x2 = tf.minimum(p_x2, g_x2)
        inter_area =  tf.multiply(tf.maximum(i_y2-i_y1, 0), tf.maximum(i_x2-i_x1, 0))
        
        # Intersection over union
        iou = inter_area / (tf.add(p_area, g_area) - inter_area)
        iou = tf.reshape(iou, [tf.shape(proposal_per_img)[0], tf.shape(gt_boxes_per_img)[0]])
        return iou
        
    
    
    def build_detection_target(self, proposals, gt_class_ids, gt_bboxes):
        '''
        
        :param proposals:       [batch_size, 2000, 4], 2000-> when training, 1000 -> when testing, Are Zero Padded
        :param gt_bboxes:       [batch_size, 100, 4], 100 -> max_gt_instances, Are Zero padded
        :param gt_class_ids:    [batch_size, 100], 100 -> max_gt_instances, Are Zero padded
        :return:
                                rois: [batch_size, MRCNN_TRAIN_ROIS_PER_IMAGE]
                                    We require equall number of +ve and -ve rois to do roi pooling and get
                                    predictions.  Using the prediction "logits" and "box" we compute the loss between
                                    the prediction and "roi_gt_class_ids" and "roi_gt_box_deltas"
                                roi_gt_class_ids: [batch_size, MRCNN_TRAIN_ROIS_PER_IMAGE], -> [batch_size, object_id]
                                    Used in Classification Loss.
                                roi_gt_box_deltas: [batch_size, MRCNN_TRAIN_ROIS_PER_IMAGE]
                                    Used in regression (bounding box) loss. Uses only positive ROI's. Proposal (ROIs)
                                    that have +ive iou with gt_bboxes
                                
        
        
        The proposals we receive from the proposal layer are :
                [num_batch, num_proposals, (y1, x1, y2, x2)]
        Also, we have ground_truth (gt_boxes).
    
        In order to build the mrcnn_loss, we would require to find equal number of +ve and -ve class and similarly and
        find intersection over union of >=0.5 and <0.5
    
        Process:
        Proposals (ROIs) are post processing on RPN. From each proposals we generate the labels of what object the proposal is most likely detecting and how good it is in detecting it. This is required becasue we need to find the targets for object classification for MRCNN layer.
            1. Remove the zero padded records from the gt_boxes, gt_class_id and select proposals only for valid "gt" instances
            2. Compute intersection over union
            3. Fetch proportional +ve and -ve boxes based on the iou
            4. Fetch roi_class_id (class_id of proposals) and roi_bboxes for +ve boxes
            5. Fetch all gt_boxes, gt_class_ids that have most IOU with each +ve Proposals

        '''
        
        
        prcntg_pos_instances = 0.33
        # prcntg_neg_instances = 0.7
    
        gt_bboxes = tf.cast(gt_bboxes, dtype=tf.float32)
        # RUN for each Image
    
        # Remove the additional padding from proposals
        non_zeros = tf.cast(tf.reduce_sum(tf.abs(proposals), axis=1), dtype=tf.bool)
        prop = tf.boolean_mask(proposals, non_zeros, name='proposals_non_zeros')
    
        # Remove the additional padding from the ground truth boxes
        non_zeros = tf.cast(tf.reduce_sum(tf.abs(gt_bboxes), axis=1), dtype=tf.bool)
        gt_boxes = tf.boolean_mask(gt_bboxes, non_zeros, name='gt_box_non_zeros')
        
        # Get teh Class ID's for the Ground Truth boxes
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name='gt_class_id_non_zeros')
        
        # Proposals and gt_bboxes are in (y1, x1, y2, x2) coordinates system, we directly get the intersection over union
        # Get the Intersection over union, iou
        iou = self.get_iou_tf(prop, gt_boxes)
        # If one proposal has iou > threshold with two objects than select teh max iou (required to make a proposal only pertaining to +ve or -ve class)
        roi_iou_max = tf.reduce_max(iou, axis=1)
        
        # Get Positive >0.5 and negative <0.3 anchors
        pos_indices = tf.where(roi_iou_max >= 0.5)[:, 0]
        neg_indices = tf.where(roi_iou_max < 0.5)[:, 0]
        
        # Get Shuffle Positive and negative indices and select only a few of them
        num_pos_inst = int(self.train_rois_per_image * prcntg_pos_instances)
        pos_indices = tf.random_shuffle(pos_indices)[:num_pos_inst]
        pos_count = tf.shape(pos_indices)[0]
        
        # For negative instances we directly dont take the 0.7% of MRCNN_TRAIN_ROIS_PER_IMAGE, because in realty the +ve instances could be very high and -ve instances could be very low. So, we compute the -ve instance could when scaled with +ve count
        neg_cnt = tf.cast((1/prcntg_pos_instances) * tf.cast(pos_count, tf.float32),
                                 tf.int32) - pos_count
        # neg_cnt2 = tf.divide(tf.cast(tf.cast(prcntg_pos_instances, tf.float32) * positive_count, dtype=tf.float32) , tf.cast(prcntg_neg_instances, dtype=tf.float32))
        neg_indices = tf.random_shuffle(neg_indices)[:neg_cnt]
        
        # Get the positive and negative proposals
        pos_rois = tf.gather(proposals, pos_indices)
        neg_rois = tf.gather(proposals, neg_indices)
        
        # Assign Positive ROIs to Ground Truth boxes
        # We can have multiple proposals(rois) intersecting with gt_boxes,
        # Inorder to get the target label for proposals, we need to get the gt_box that the proposal is defining (intersecting) the most.
        # roi_gt_box_assignment are the indices of the proposals that intersects with most with each gt_boxes, This case could even output only one proposal
        pos_iou = tf.gather(iou, pos_indices)
        roi_gt_box_assignment = tf.argmax(pos_iou, axis=1)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        
        # Refine boxes to adjust the predicted boxes to gt_box coordinates
        roi_gt_box_deltas = self.box_refinement_tf(pos_rois, roi_gt_boxes)
        roi_gt_box_deltas /= self.box_stddev
        
        
        # Concat and pad all generated Tensors to ease processing
        rois = tf.concat([pos_rois, neg_rois], axis=0)
        neg_rois_cnt = tf.shape(neg_rois)[0]
        num_pad = tf.maximum(self.train_rois_per_image - tf.shape(rois)[0], 0)
        self.rois = tf.pad(rois, [(0,num_pad), (0,0)])
        self.roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0,num_pad+neg_rois_cnt)])
        self.roi_gt_box_deltas = tf.pad(roi_gt_box_deltas, [(0, num_pad + neg_rois_cnt), (0, 0)])

        # Reshape, to convert into nd array
        self.roi_gt_class_ids = tf.reshape(self.roi_gt_class_ids, shape=[1,tf.shape(self.roi_gt_class_ids)[0]])
        
        if self.DEBUG:
            self.debug_dict = dict(a=prop, b=gt_boxes, c=gt_class_ids, d=iou, e=pos_rois,
                    f=neg_rois, g=pos_iou, h=roi_gt_box_assignment, i=roi_gt_class_ids,
                    j=self.roi_gt_class_ids, k=self.roi_gt_box_deltas, l=self.rois)
    
    def get_target_rois(self):
        return self.rois, self.roi_gt_class_ids, self.roi_gt_box_deltas

    def debug_outputs(self):
        return self.debug_dict
                



class PreprareTrainData():
    def __init__(self, conf, dataset):
        # TODO: Handle the case where the output data has no class, but only background.
        
        self.image_min_dim = conf.IMAGE_MIN_DIM
        self.image_max_dim = conf.IMAGE_MAX_DIM
        self.min_scale = conf.IMAGE_MIN_SCALE
        self.resize_mode = conf.IMAGE_RESIZE_MODE
        self.max_rpn_targets = conf.RPN_TRAIN_ANCHORS_PER_IMAGE
        self.bbox_std_dev = conf.RPN_BBOX_STDDEV
        self.max_gt_objects_per_image = conf.MAX_GT_OBJECTS
        
        self.conf = conf
        # print (self.image_min_dim, self.image_max_dim, self.min_scale, self.resize_mode)
        
        self.dataset = dataset

        # Get Anchors to compute overlap between anchors and groundtruth boxes
        # Also compute the area of the anchors, so that we dont have to compute the both the anchors and the
        # area in every batch iteration
        feature_shapes = utils.get_resnet_stage_shapes(self.conf, self.conf.IMAGE_SHAPE)
        self.anchors = utils.gen_anchors_for_train(self.conf.RPN_ANCHOR_SCALES,
                                              self.conf.RPN_ANCHOR_RATIOS,
                                              feature_shapes,
                                              self.conf.RESNET_STRIDES,
                                              self.conf.RPN_ANCHOR_STRIDE)
        
        # print(feature_shapes)
        print('Anchors Max Min length: ', np.max(self.anchors), np.min(self.anchors))
        print ('Anchor shape: ', self.anchors.shape)
        
        self.anchor_area = (self.anchors[:,2] - self.anchors[:,0]) * (self.anchors[:,3] - self.anchors[:,1])
        print ('Achor Area: ', self.anchor_area.shape)
    
    def extract_bboxes(self, mask):
        '''
        :param mask: [height, width, num_objects]
        :return:     Given a mask, outputs a bounding box with "lower left" and "upper right" coordinates
        '''
        bboxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        shift = [0, 0, 1, 1]
        for i in range(0, mask.shape[-1]):
            msk = mask[:, :, i]
            horizontal_coord = np.where(np.any(msk, axis=0))[0]
            vertical_coord = np.where(np.any(msk, axis=1))[0]
            
            if len(horizontal_coord) >= 0 and len(vertical_coord) >= 0:
                x1, x2 = horizontal_coord[[0, -1]]
                y1, y2 = vertical_coord[[0, -1]]
                
                bbox = np.array([y1, x1, y2, x2]) + shift
            else:
                bbox = [0, 0, 0, 0]
            bboxes[i] = bbox
        return bboxes.astype(np.int32)
    
    def get_ground_truth_data(self, image_id):
        image = self.dataset.get_image(image_id=image_id)
        gt_mask = self.dataset.get_object_mask(image_id=image_id)
        gt_class_ids = self.dataset.get_class_labels(image_id=image_id)
        active_class_ids = np.zeros(self.dataset.num_classes, dtype=np.int32)
        active_class_ids[gt_class_ids] = 1
        original_image_shape = image.shape
        
        image, image_window, scale, padding = utils.resize_image(
                image, min_dim=self.image_min_dim,
                max_dim=self.image_max_dim,
                min_scale=self.min_scale,
                mode=self.resize_mode
        )
        gt_mask = utils.resize_mask(gt_mask, scale, padding)
        gt_bboxes = self.extract_bboxes(gt_mask)
        image_metas = compose_image_meta(image_id, original_image_shape, image.shape,
                                             image_window, scale, active_class_ids)
        print('active_class_ids ', active_class_ids)
        # print ('poweurweuriweuriwe, ', gt_class_ids)
        return image, gt_mask, gt_class_ids, gt_bboxes, image_metas


    def build_rpn_targets(self, batch_gt_boxes):
        ''' Building RPN target for classification
        
        RPN produces two outputs 1) rpn_bboxes, 2) rpn_class_probs.
        
        Suppose we have a feature_maps: [32,32], [16,16], [8,8], [4,4], [2,2] and num_anchors_per_pixel=3
        then, Anchor shape = [32x32x3 + 16x16x3 + .....+ 2*2*3, 4] = [4092, 4],
        then, we need to find which anchors have <0.3 iou and >0.7 iou with the bounding box.
        this is required because only then we can know which anchors are -ve and which are positive.
        Also, RPN would output a rpn_bbox of shape [4092, 4] and we need to build a
        loss function for RPN module. RPN produces two outputs, therefore we would have two losses.
        
        1. Classification loss, penalize if the model does bad at scoring the bounding boxes
        2. Regression loss: penalizes is the classified bbox has bad [c_x, c_y, log(h), log(w)]
        
        :return:
        '''
        
        ##### CLASSIFICATION PART
        rpn_target_class = np.zeros([self.anchors.shape[0]], dtype='int32' )
        
        # print (batch_gt_boxes.shape)
        batch_gt_area = (batch_gt_boxes[:,2] - batch_gt_boxes[:,0]) * (batch_gt_boxes[:,3] - batch_gt_boxes[:,1])
        # print (batch_gt_area.shape)
        overlaps = np.zeros(shape=(batch_gt_boxes.shape[0], self.anchors.shape[0]))
        
        # print ('')
        
        for i in range(0, overlaps.shape[0]):
            overlaps[i, :] = utils.intersection_over_union(
                    batch_gt_boxes[i], self.anchors, batch_gt_area[i], self.anchor_area
            )
        overlaps = overlaps.T
        # print(overlaps.shape, np.max(overlaps), np.min(overlaps))
        # print(overlaps)
    
        # Apply conditions,
        # When bounding_box iou anchor > 0.7 class 1
        # When bounding_box iou anchor < 0.3 class 0
        # Else doesn't matter
        # One important information to note:
        # one image may have multiple anchors, say anchor 1 iou object 1 is >0.7 and iou with object 2 <0.3
        # So, if we select positives anchors first then anchor 1 might not be selected because after object 1
        # object 2 will disqualify anchor1. Therefore we should first select negative anchors.
        # Moreover, if anchor1 iou with object 1 is 0.7 and object2 is 0.9 then we must choose the top score
        # for the anchor.
        anchor_iou_max_idx = np.argmax(overlaps, axis=1)  # Choose the highest score per anchor
        anchor_iou_max_score = overlaps[np.arange(len(overlaps)), anchor_iou_max_idx]
        # print(len(anchor_iou_max_score), anchor_iou_max_score)
        # print ('')
        
        # COND 1: Set rpn_target_class to -1 where the iou is < 0.3
        rpn_target_class[(anchor_iou_max_score < 0.3)] = -1
        # print (rpn_target_class)
        
        # COND 2: When all anchors are <0.7 to ground truth, we still choose the best anchor to perform the training.
        gt_iou_max_idx = np.argmax(overlaps, axis=0)
        rpn_target_class[gt_iou_max_idx] = 1
        
        # COND 3: TODO: Handle the condition when multiple anchors can have the same IOU
        
        # COND 4: Set rpn_target_class = 1 where the iou is > 0.7
        rpn_target_class[(anchor_iou_max_score >= 0.7)] = 1
        # print(rpn_target_class)
        
        # The Positive and negative anchors should be balanced, otherwise the model may get biased.
        # TODO: Even though the max_rpn_target is 256, it may be the case that we were able to find 128 -ve class but only 1 +ve class. This would highly bias the network, check if this is the case even when using pretrained resnet weights.
        
        idx = np.where(rpn_target_class == 1)[0]
        print('pos_idx, ', len(idx))
        extra = len(idx) - self.max_rpn_targets // 2
        # print('extra 1 ', extra)
        if (extra) > 0:
            rpn_target_class[np.random.choice(idx, extra, replace=False)] = 0

        idx = np.where(rpn_target_class == -1)[0]
        print('neg_idx, ', len(idx), self.max_rpn_targets // 2)
        extra = len(idx) - self.max_rpn_targets // 2
        if extra > 0:
            rpn_target_class[np.random.choice(idx, extra, replace=False)] = 0
        # print('extra -1 ', extra)
        
        # REGRESSION PART:
        rpn_target_bbox = np.zeros((self.max_rpn_targets, 4))
        pos_idx = np.where(rpn_target_class == 1)[0]
        for i, (idx, anchor_box) in enumerate(zip(pos_idx, self.anchors[pos_idx])):
            # Convert bbox to scaled and shifted version
            gt_box = batch_gt_boxes[anchor_iou_max_idx[idx]] # Select class 1 from gt boxes with high iou score
            # Get center, h, w of anchor boxes
            ah = anchor_box[2] - anchor_box[0]
            aw = anchor_box[3] - anchor_box[1]
            ac_y = anchor_box[0] + 0.5*ah
            ac_x = anchor_box[1] + 0.5*aw

            # Get center, h, w of anchor boxes
            gth = gt_box[2] - gt_box[0]
            gtw = gt_box[3] - gt_box[1]
            gtc_y = gt_box[0] + 0.5 * gth
            gtc_x = gt_box[1] + 0.5 * gtw

            rpn_target_bbox[i] = [
                (gtc_y - ac_y) / ah,       # Distance between anchor bbox and gt_box
                (gtc_x - ac_x) / aw,
                np.log(gth / ah),
                np.log(gtw / aw),
            ]
            
            # Normalize
            rpn_target_bbox[i] /= self.bbox_std_dev

        return rpn_target_class, rpn_target_bbox
        
    def build_mrcnn_targets(self, batch_gt_boxes, batch_gt_class):
        ''' What actually goes inside it
        
        The Mask RCNN module outputs 1) get_mrcnn_class_probs and 2) get_mrcnn_bbox.
        So there are again two losses we need to consider
        1) Classification loss : 81 classes including one for foreground, for shapes we have only 4 classes
            We don't perform the classification loss here, but we use it to extract bounding boxes.
        2) Regression loss: The bounding box loss just like we did in the rpn module
        
        :param proposals:
        :return:
        '''
        # TODO: This module is only required if we plan to train the model without training the RPN Head.
    
        
    def get_data(self, image_ids):
        batch_size = len(image_ids)
        
        for num, img_id in enumerate(image_ids):
            image, gt_mask, gt_class_id, gt_bbox, image_meta = self.get_ground_truth_data(img_id)
            
            # GET RPN TARGETS
            rpn_target_class, rpn_target_bbox = self.build_rpn_targets(gt_bbox)
            print('+ve class count ', len(np.where(rpn_target_class == 1)[0]))
            print('-ve class count ', len(np.where(rpn_target_class == -1)[0]))
            print('neutral class count ', len(np.where(rpn_target_class == 0)[0]))
            # print ('rpn_target_classrpn_target_class ', rpn_target_class.shape)

            # TODO: If gt_box exceeds the number of maximum allowed then select the top best

            # We preinitialize each nd_array such that we save compute by not appending for every image
            if num == 0:
                batch_images = np.zeros((batch_size,) + tuple(image.shape), dtype=np.float32)
                
                # Note gt_masks, gt_class_id and gt_boxes count may differ for each image so
                # We handle them seperately
                batch_gt_masks = np.zeros((batch_size, gt_mask.shape[0], gt_mask.shape[0], self.max_gt_objects_per_image), dtype=gt_mask.dtype)
                batch_gt_class_ids = np.zeros((batch_size, self.max_gt_objects_per_image),dtype=gt_class_id.dtype)
                batch_gt_bboxes = np.zeros((batch_size, self.max_gt_objects_per_image, 4), dtype=gt_bbox.dtype)
                
                batch_image_metas = np.zeros((batch_size,) + tuple(image_meta.shape), dtype=image_meta.dtype)
                batch_rpn_target_class = np.zeros([batch_size, self.anchors.shape[0], 1], dtype=rpn_target_class.dtype)
                batch_rpn_target_bbox = np.zeros((batch_size,) + tuple(rpn_target_bbox.shape), dtype=rpn_target_bbox.dtype)
            
            batch_images[num] = image
            batch_gt_masks[num,:,:, :gt_mask.shape[-1]] = gt_mask
            batch_gt_class_ids[num, :gt_class_id.shape[0]] = gt_class_id
            batch_gt_bboxes[num, :gt_bbox.shape[0]] = gt_bbox
            batch_image_metas[num] = image_meta
            batch_rpn_target_class[num] = rpn_target_class[:, np.newaxis]
            batch_rpn_target_bbox[num] = rpn_target_bbox
            
        print('batch_images ', batch_images.shape)
        print('batch_gt_masks ', batch_gt_masks.shape)
        print('batch_gt_class_ids ', batch_gt_class_ids.shape)
        print('batch_gt_bboxes ', batch_gt_bboxes.shape)
        print('batch_image_metas ', batch_image_metas.shape)
        print('batch_rpn_target_class ', batch_rpn_target_class.shape)
        print('batch_rpn_target_bbox ', batch_rpn_target_bbox.shape)
        
        return (dict(batch_images=batch_images, batch_image_metas=batch_image_metas, batch_gt_masks=batch_gt_masks,
                     batch_gt_class_ids=batch_gt_class_ids, batch_gt_bboxes=batch_gt_bboxes,
                     batch_rpn_target_class=batch_rpn_target_class, batch_rpn_target_bbox=batch_rpn_target_bbox))


def debug():
    from MaskRCNN.config import config as conf
    list_of_images = [np.random.random((800,1024,3)), np.random.random((800,1024,3))]
    list_of_image_ids = [1,2]
    process_images(conf, list_of_images, list_of_image_ids)
    
# debug()


