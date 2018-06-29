
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







def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

class PreprareTrainData():
    def __init__(self, conf, dataset):
        self.image_min_dim = conf.IMAGE_MIN_DIM
        self.image_max_dim = conf.IMAGE_MAX_DIM
        self.min_scale = conf.IMAGE_MIN_SCALE
        self.resize_mode = conf.IMAGE_RESIZE_MODE
        
        self.conf = conf
        # print (self.image_min_dim, self.image_max_dim, self.min_scale, self.resize_mode)
        
        self.dataset = dataset

        # Get Anchors to compute overlap between anchors and groundtruth boxes
        # Also compute the area of the anchors, so that we dont have to compute the both the anchors and the
        # area in every batch iteration
        feature_shapes = utils.get_resnet_stage_shapes(self.conf, self.conf.IMAGE_SHAPE)
        self.anchors = utils.gen_anchors_fot_train(self.conf.RPN_ANCHOR_SCALES,
                                              self.conf.RPN_ANCHOR_RATIOS,
                                              feature_shapes,
                                              self.conf.RESNET_STRIDES,
                                              self.conf.RPN_ANCHOR_STRIDE)
        
        # print(feature_shapes)
        print('Anchors Max Min length: ', np.max(self.anchors), np.min(self.anchors))
        
        self.anchor_area = (self.anchors[:,2] - self.anchors[:,0]) * (self.anchors[:,3] - self.anchors[:,1])
        print ('Achor Area: ', self.anchor_area.shape)
    
    def extract_bboxes(self, mask):
        '''
        :param mask: [height, width, num_objects]
        :return:     Given a mask outputs a bounding box with "lower left" and "upper right" coordinites
        '''
        bboxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        shift = [0, 0, 1, 1]
        for i in range(0, mask.shape[-1]):
            msk = mask[:, :, i]
            horizontal_coord = np.where(np.any(msk, axis=0))[0]
            vertical_coord = np.where(np.any(msk, axis=1))[0]
            
            if len(horizontal_coord) >= 0 and len(horizontal_coord) >= 0:
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
        
        image, image_window, scale, padding = utils.resize_image(image, min_dim=self.image_min_dim,
                                                                 max_dim=self.image_max_dim,
                                                                 min_scale=self.min_scale,
                                                                 mode=self.resize_mode)
        gt_mask = utils.resize_mask(gt_mask, scale, padding)
        gt_bboxes = self.extract_bboxes(gt_mask)
        image_metas = compose_image_meta(image_id, original_image_shape, image.shape,
                                             image_window, scale, active_class_ids)
        return image, gt_mask, gt_class_ids, gt_bboxes, image_metas

    def build_rpn_targets(self, batch_gt_boxes):
        ''' Building RPN target for classification
        
        RPN produces two outputs 1) rpn_bboxes, 2) rpn_class_probs. The rpn_bboxes
        
        Suppose we have a feature_maps: [32,32], [16,16], [8,8], [4,4], [2,2] and num_anchors_per_pixel=3
        then, Anchor shape = [32x32x3 + 16x16x3 + .....+ 2*2*3, 4] = [4092, 4],
        then, we need to find which anchors have <0.3 iou and >0.7 iou with the bounding box.
        this is required because only then we can know which anchors are -ve and which are positive.
        becasue RPN would output a rpn_bbox of shape [4092, 4] and we need to build a
        loss function for RPN module
        
        :return:
        '''
        overlaps = np.zeros(shape=(len(batch_gt_boxes), self.anchors.shape[0]))
        for i, boxes in enumerate(batch_gt_boxes):
            print (boxes)
            box_area = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
            overlaps[i,:] = compute_iou(
                    boxes, self.anchors, box_area, self.anchor_area
            )
            print(np.max(overlaps[i,:]))
            
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
        print (overlaps.shape)
        anchor_iou_max_idx = np.argmax(overlaps, axis=1)  # Choose the highest score per anchor
        print (len(anchor_iou_max_idx), anchor_iou_max_idx)
        anchor_iou_max_score = overlaps[np.arange(len(overlaps)), anchor_iou_max_idx]
        print(len(anchor_iou_max_score), anchor_iou_max_score)
        # anchor_iou_idx = overlaps[np.arange(overlaps.shape[0]), anchor_iou_idx]
        # print ('')
        # print(len(anchor_iou_idx), anchor_iou_idx)
        
    def get_data(self, image_ids):
        batch_images = []
        batch_gt_masks = []
        batch_gt_class_ids = []
        batch_gt_bboxes = []
        batch_image_metas = []

        for img_id in image_ids:
            image, gt_mask, gt_class_id, gt_bbox, image_meta = self.get_ground_truth_data(img_id)

            # GET RPN TARGETS
            self.build_rpn_targets(gt_bbox)
            
            batch_images.append(image)
            batch_gt_masks.append(gt_mask)
            batch_gt_class_ids.append(batch_gt_class_ids)
            batch_gt_bboxes.append(gt_bbox)
            
            break
            

        # batch_images = np.stack(batch_images, axis=0)
        # return (batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes,
        #         np.array(batch_image_metas).astype(np.int32))


def debug():
    from MaskRCNN.config import config as conf
    list_of_images = [np.random.random((800,1024,3)), np.random.random((800,1024,3))]
    list_of_image_ids = [1,2]
    process_images(conf, list_of_images, list_of_image_ids)
    
# debug()


