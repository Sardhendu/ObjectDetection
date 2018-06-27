
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
        
        self.conf = conf
        # print (self.image_min_dim, self.image_max_dim, self.min_scale, self.resize_mode)
        
        self.dataset = dataset
    
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

    def build_rpn_targets(self, batch_size):
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
        pass
        
    def get_data(self, image_ids):
        batch_images = []
        batch_gt_masks = []
        batch_gt_class_ids = []
        batch_gt_bboxes = []
        batch_image_metas = []

        # Get Anchors to compute overlap between anchors and groundtruth boxes
        feature_shapes = utils.get_resnet_stage_shapes(self.conf, self.conf.IMAGE_SHAPE)
        anchors = utils.gen_anchors_fot_train(self.conf.RPN_ANCHOR_SCALES,
                                              self.conf.RPN_ANCHOR_RATIOS,
                                              feature_shapes,
                                              self.conf.RESNET_STRIDES,
                                              self.conf.RPN_ANCHOR_STRIDE)
        print(feature_shapes)
        print(anchors.shape)

        for img_id in image_ids:
            image, gt_mask, gt_class_id, gt_bbox, image_meta = self.get_ground_truth_data(img_id)

            batch_images.append(image)
            batch_gt_masks.append(gt_mask)
            batch_gt_class_ids.append(batch_gt_class_ids)
            batch_gt_bboxes.append(gt_bbox)

        batch_images = np.stack(batch_images, axis=0)
        return (batch_images, batch_gt_masks, batch_gt_class_ids, batch_gt_bboxes,
                np.array(batch_image_metas).astype(np.int32))


def debug():
    from MaskRCNN.config import config as conf
    list_of_images = [np.random.random((800,1024,3)), np.random.random((800,1024,3))]
    list_of_image_ids = [1,2]
    process_images(conf, list_of_images, list_of_image_ids)
    
# debug()


