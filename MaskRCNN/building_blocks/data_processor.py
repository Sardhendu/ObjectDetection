
'''
The RCNN module takes input as a certain format. For instance the

'''

from skimage.transform import resize
import numpy as np
import math
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


class PreprareTrainData():
    def __init__(self, conf, dataset):
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
            # Convert bbox to scalled and shifted version
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


