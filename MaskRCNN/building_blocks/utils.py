


import numpy as np

from skimage import transform
import scipy
import logging



logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


def intersection_over_union(box, reference_boxes, box_area, reference_box_areas):
    intersection_y1 = np.maximum(box[0], reference_boxes[:, 0])
    intersection_x1 = np.maximum(box[1], reference_boxes[:, 1])
    intersection_y2 = np.minimum(box[2], reference_boxes[:, 2])
    intersection_x2 = np.minimum(box[3], reference_boxes[:, 3])
    
    intersection_area = (np.maximum((intersection_y2 - intersection_y1), 0) *
                         np.maximum((intersection_x2 - intersection_x1), 0))
    
    union_area = box_area + reference_box_areas[:] - intersection_area
    iou = intersection_area / union_area
    return iou


def non_max_supression(boxes, scores, threshold):
    y2 = boxes[:, 2]
    y1 = boxes[:, 0]
    x2 = boxes[:, 3]
    x1 = boxes[:, 1]
    box_areas = (y2 - y1) * (x2 - x1)
    
    idx = scores.argsort()[::-1]
    pick = []
    while len(idx) > 0:
        i = idx[0]
        pick.append(i)
        
        # Compute the iou between the picked box and the rest
        iou = intersection_over_union(boxes[i], boxes[idx[1:]], box_areas[i], box_areas[idx[1:]])
        
        # If iou exceeds the threshold then then we say that the picked box and few of the rest have
        # high IOU, hence we should remove the such boxes.
        delete_idx = np.where(iou > threshold)[0] + 1  # +1 because the current box is at 0th porision
        
        idx = np.delete(idx, delete_idx)  # remove the idx from reference set
        idx = np.delete(idx, 0)  # remove the picked idx, since we dont want to run it again
    return np.array(pick, dtype=np.int32)



def normalize_image(images, mean_pixels):
    """
    Expects an RGB image (or array of images) and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - mean_pixels


def resize_image(image, min_dim, max_dim, min_scale, mode='square'):
    ''' Adjusting image shape for processing throught the NET

    To Note:
    1. the scale can not be less than min_scale i.e (0)
    2. the scale can not exceed the max_scale i.e (max_dim/max_image_dim)
    3.
    :param image:       Can be of any dimension and resolution
    :param min_dim:     min_dim mostly 800
    :param max_dim:     max_dim
    :param min_scale:   0
    :param mode:        square [h = w]
    :return:
    '''
    h, w = image.shape[0:2]
    
    # Scale up not down, based on the min dimension of the image
    scale = max(1, min_dim / min(h, w))
    
    # Ensure the scale is greater or equal to the minimum scale
    scale = max(scale, min_scale) if min_scale else scale
    
    # Ensure that the scale is smaller or equal to the maximum scale
    max_scale = max_dim / max(h, w)
    scale = min(scale, max_scale)
    
    # If the scale is greater than 1 then perform bilinear interpolation to increase the size of h and w by the scale
    if scale != 1:
        image = transform.resize(image, (round(h * scale), round(w * scale)), order=1, mode='constant',
                                 preserve_range=True)
    
    # Image may still be smaller than teh desired size [max_dim, max_dim], we should perform zero padding if so.
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    image_window = (top_pad, left_pad, h + top_pad, w + left_pad)
    
    return image.astype(image.dtype), image_window, scale, padding


def resize_mask(mask, scale, padding):
    '''
    :param mask:        [height, width, num_objects]
    :param scale:       float value
    :param padding:     [(),(),(),()]
    :return:            Zoom, minimize or pad based on the scale and padding values
    '''
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def get_resnet_stage_shapes(conf, image_shape):
    '''
    Getting RESNET Pyramid Stage Shapes
    
    :param conf:
    :param image_shape:
    :return:
    
    What is aimed of it:
    In our assumed Resnet 101 model we take a input of 1024x1024. and each stage of pyramid layer we reduce the feature
    map. For instance,
        Stage P1: Feature map = 256x256
        Stage P2: Feature map = 128x128
        Stage P3: Feature map = 64x64
        Stage P4: Feature map = 32x32
        Stage P5: Feature map = 16x16
        
    since we run our anchors on top of these feature maps, we calculate the output shape of these Pyramid-stages to
    get our anchor shape.
    '''
    return np.array(
            [[int(np.ceil(image_shape[0] / stride)),
              int(np.ceil(image_shape[1] / stride))]
             for stride in conf.RESNET_STRIDES])


def norm_boxes(box, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])  # [y1, x1, y2, x2]
    return np.divide((box - shift), scale).astype(np.float32)

def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    print('RUNNING utils (denorm_boxes)......................')
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def generate_anchors_for_feature_map(scales, ratios, feature_map_shapes, feature_map_strides, anchor_strides):
    """
    Generates anchors for each feature map
    
    :param scales:                  1D array of anchor sizes in pixels. Example: [32, 64, 128]
    :param ratios:                  1D array of ratios [0.5, 1, 2]
    :param feature_map_shapes:       Shape of each feature map from teh FPN network
    :param feature_map_stride:      1D array: Number of strides require to convolve the image into each
                                    feature map Normally [4, 8,16, 32, 64]
    :param anchor_strides:          Normally 1 (We require anchor at each position of the feature map)
    
    scales:
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
        
    What is it aimed for:
    1. for each feature map we need to compute three different types of anchor
        --> A Square anchor (ratio h:w = 1:1)
        --> A horizontal rectangular anchor (ratio h:w = 0.5:1)
        --> A vertical rectangular anchor (ratio h:w = 1:2)
        Example: If feature_map_shapes = [64x64], scale = 128 and feature_map_stride = 16
                then anc1=[128 x 128] anc2=[181.02 x 90.51] anc3=[90.51 x 181.02]
    
    2. Now these three anchors developed in the above step would have to slide through every pixel with a stride 16.
        So we have a stride of 16, therefore,
        Box center for each anchor at [0,0 ,     0,16,     0,32,   ............]
                                      [16,0,     16,16,    16,32,  ............]
                                      .                                             x  3 (for three anchors)
                                      .
                                      [1008,0    1008,16   1008,32, ...........]
        TOTAL =
            center x = 64x64x3 = 4096x3
            center y = 64x64x3 = 4096x3
            Stacked flattened center_xy = 4096x3x2 = 12288, 2
        
    4. Similar to the centers we need to find a bounding box. We have three anchors with width and height known for
    each anchor
        Here, for each 16th pixel position we would have 3 anchors and one bounding box each anchor
        TOTAL =
            box height = 64x64x3 = 4096x3
            box width = 64x64x3 = 4096x3
            Stacked flattened box_hw = 4096x3x2 = 12288, 2
            
    5. Convert the box centers, height and width to get the corner coordinates.
        For example,
        If center_x = 0, center_y = 0, height = 180, width = 90, then
        
                                |----90----|
        
          --                     __________(45, 90)
          |                      |        |
          |                      |        |
          |                      |        |
         180                     |   .    |
          |                      | (0,0)  |
          |                      |        |
          |                      |        |
          --            (-45,-90)----------
          
          Using Step 3 and Step 4 we output a [12288 x 4] dimensional output.
        
    
         
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    
    # # Enumerate shifts in feature space
    print ('asdasdsad ', anchor_strides, feature_map_shapes[0])
    print(np.arange(0, feature_map_shapes[0], anchor_strides))
    shifts_y = np.arange(0, feature_map_shapes[0], anchor_strides) * feature_map_strides
    shifts_x = np.arange(0, feature_map_shapes[1], anchor_strides) * feature_map_strides
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    
    # # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    logging.info('Anchors: Box width shape = %s', str(box_widths.shape))
    logging.info('Anchors: Box height shape = %s', str(box_heights.shape))
    logging.info('Anchors: Box center_x shape = %s', str(box_centers_x.shape))
    logging.info('Anchors: Box center_y shape = %s', str(box_centers_y.shape))

    # # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])


    # # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    logging.info('Anchors: Stacked Box corner coordinates shape = %s', str(boxes.shape))

    return boxes




def gen_anchors(image_shape, batch_size, scales, ratios, feature_map_shapes, feature_map_strides, anchor_strides):
    """
    Create anchor boxes for each feature_map of pyramid stage and concat them
    """
    anchors = []
    for i in range(0,len(scales)):
            # print ('running for ', scales[i], feature_map_shapes[i], feature_map_strides[i])
        logging.info('Anchors: running for..... scales=%s, feature_map_shapes=%s, feature_map_strides=%s',
                     str(scales[i]), str(feature_map_shapes[i]), str(feature_map_strides[i]))
        anchors.append(generate_anchors_for_feature_map(scales[i], ratios, feature_map_shapes[i], feature_map_strides[i], anchor_strides))
    anchors = np.concatenate(anchors, axis=0)
    logging.info('Anchors: concatenated for each stage: shape = %s', str(anchors.shape))
    anchors = np.broadcast_to(anchors, (batch_size,) + anchors.shape)
    logging.info('Anchors: Broadcast to num_batches: shape = %s', str(anchors.shape))

    # Normalize Anchors
    anchors = norm_boxes(anchors, shape=image_shape[:2])
    return anchors



def gen_anchors_for_train(scales, ratios, feature_map_shapes, feature_map_strides, anchor_strides):
    """
    Create anchor boxes for each feature_map of pyramid stage and concat them
    """
    anchors = []
    for i in range(0,len(scales)):
            # print ('running for ', scales[i], feature_map_shapes[i], feature_map_strides[i])
        logging.info('Anchors: running for..... scales=%s, feature_map_shapes=%s, feature_map_strides=%s',
                     str(scales[i]), str(feature_map_shapes[i]), str(feature_map_strides[i]))
        anchors.append(generate_anchors_for_feature_map(scales[i], ratios, feature_map_shapes[i], feature_map_strides[i], anchor_strides))
    anchors = np.concatenate(anchors, axis=0)
    logging.info('Anchors: concatenated for each stage: shape = %s', str(anchors.shape))
    return anchors
    


def gen_random_mrcnn_boxes():
    '''
    MRCNN module is very similar to the the RPN module, Just like we needed anchors in RPN module, we would
    have to generate boxes here such that we can aid in performing regression and classification loss.
    
    The mrcnn_box outputs tensor of (1, 2000, 81, 4),
    :return:
    '''
    
    # TODO: This module is only required if we plan to train the model without training the RPN Head.
    #
    pass





#
#
# import skimage
#
#
#
#
# def compute_iou(box, boxes, box_area, boxes_area):
#     """Calculates IoU of the given box with the array of the given boxes.
#     box: 1D vector [y1, x1, y2, x2]
#     boxes: [boxes_count, (y1, x1, y2, x2)]
#     box_area: float. the area of 'box'
#     boxes_area: array of length boxes_count.
#
#     Note: the areas are passed in rather than calculated here for
#           efficency. Calculate once in the caller to avoid duplicate work.
#     """
#     # Calculate intersection areas
#     y1 = np.maximum(box[0], boxes[:, 0])
#     y2 = np.minimum(box[2], boxes[:, 2])
#     x1 = np.maximum(box[1], boxes[:, 1])
#     x2 = np.minimum(box[3], boxes[:, 3])
#     intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
#     union = box_area + boxes_area[:] - intersection[:]
#     iou = intersection / union
#     return iou
#
#
# def compute_overlaps(boxes1, boxes2):
#     """Computes IoU overlaps between two sets of boxes.
#     boxes1, boxes2: [N, (y1, x1, y2, x2)].
#
#     For better performance, pass the largest set first and the smaller second.
#     """
#     # Areas of anchors and GT boxes
#     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
#
#     # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
#     # Each cell contains the IoU value.
#     overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
#     for i in range(overlaps.shape[1]):
#         box2 = boxes2[i]
#         overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
#     return overlaps
#
#
# def compute_overlaps_masks(masks1, masks2):
#     '''Computes IoU overlaps between two sets of masks.
#     masks1, masks2: [Height, Width, instances]
#     '''
#     # flatten masks
#     masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
#     masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
#     area1 = np.sum(masks1, axis=0)
#     area2 = np.sum(masks2, axis=0)
#
#     # intersections and union
#     intersections = np.dot(masks1.T, masks2)
#     union = area1[:, None] + area2[None, :] - intersections
#     overlaps = intersections / union
#
#     return overlaps
#
#
# def non_max_suppression(boxes, scores, threshold):
#     """Performs non-maximum supression and returns indicies of kept boxes.
#     boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
#     scores: 1-D array of box scores.
#     threshold: Float. IoU threshold to use for filtering.
#     """
#     assert boxes.shape[0] > 0
#     if boxes.dtype.kind != "f":
#         boxes = boxes.astype(np.float32)
#
#     # Compute box areas
#     y1 = boxes[:, 0]
#     x1 = boxes[:, 1]
#     y2 = boxes[:, 2]
#     x2 = boxes[:, 3]
#     area = (y2 - y1) * (x2 - x1)
#
#     # Get indicies of boxes sorted by scores (highest first)
#     ixs = scores.argsort()[::-1]
#
#     pick = []
#     while len(ixs) > 0:
#         # Pick top box and add its index to the list
#         i = ixs[0]
#         pick.append(i)
#         # Compute IoU of the picked box with the rest
#         iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
#         # Identify boxes with IoU over the threshold. This
#         # returns indicies into ixs[1:], so add 1 to get
#         # indicies into ixs.
#         remove_ixs = np.where(iou > threshold)[0] + 1
#         # Remove indicies of the picked and overlapped boxes.
#         ixs = np.delete(ixs, remove_ixs)
#         ixs = np.delete(ixs, 0)
#     return np.array(pick, dtype=np.int32)
#
#
#
# class Dataset(object):
#     """The base class for dataset classes.
#     To use it, create a new class that adds functions specific to the dataset
#     you want to use. For example:
#
#     class CatsAndDogsDataset(Dataset):
#         def load_cats_and_dogs(self):
#             ...
#         def load_mask(self, image_id):
#             ...
#         def image_reference(self, image_id):
#             ...
#
#     See COCODataset and ShapesDataset as examples.
#     """
#
#     def __init__(self, class_map=None):
#         self._image_ids = []
#         self.image_info = []
#         # Background is always the first class
#         self.class_info = [{"source": "", "id": 0, "name": "BG"}]
#         self.source_class_ids = {}
#
#     def add_class(self, source, class_id, class_name):
#         assert "." not in source, "Source name cannot contain a dot"
#         # Does the class exist already?
#         for info in self.class_info:
#             if info['source'] == source and info["id"] == class_id:
#                 # source.class_id combination already available, skip
#                 return
#         # Add the class
#         self.class_info.append({
#             "source": source,
#             "id": class_id,
#             "name": class_name,
#         })
#
#     def add_image(self, source, image_id, path, **kwargs):
#         image_info = {
#             "id": image_id,
#             "source": source,
#             "path": path,
#         }
#         image_info.update(kwargs)
#         self.image_info.append(image_info)
#
#     def image_reference(self, image_id):
#         """Return a link to the image in its source Website or details about
#         the image that help looking it up or debugging it.
#
#         Override for your dataset, but pass to this function
#         if you encounter images not in your dataset.
#         """
#         return ""
#
#     def prepare(self, class_map=None):
#         """Prepares the Dataset class for use.
#
#         TODO: class map is not supported yet. When done, it should handle mapping
#               classes from different datasets to the same class ID.
#         """
#         print ('prepare ...........')
#         def clean_name(name):
#             """Returns a shorter version of object names for cleaner display."""
#             return ",".join(name.split(",")[:1])
#
#         # Build (or rebuild) everything else from the info dicts.
#         self.num_classes = len(self.class_info)
#         self.class_ids = np.arange(self.num_classes)
#         self.class_names = [clean_name(c["name"]) for c in self.class_info]
#         self.num_images = len(self.image_info)
#         self._image_ids = np.arange(self.num_images)
#
#         # Mapping from source class and image IDs to internal IDs
#         self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
#                                       for info, id in zip(self.class_info, self.class_ids)}
#         self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
#                                       for info, id in zip(self.image_info, self.image_ids)}
#
#         # Map sources to class_ids they support
#         self.sources = list(set([i['source'] for i in self.class_info]))
#         self.source_class_ids = {}
#         # Loop over datasets
#         for source in self.sources:
#             self.source_class_ids[source] = []
#             # Find classes that belong to this dataset
#             for i, info in enumerate(self.class_info):
#                 # Include BG class in all datasets
#                 if i == 0 or source == info['source']:
#                     self.source_class_ids[source].append(i)
#
#     def map_source_class_id(self, source_class_id):
#         """Takes a source class ID and returns the int class ID assigned to it.
#
#         For example:
#         dataset.map_source_class_id("coco.12") -> 23
#         """
#         print('map_source_class_id ..............')
#         return self.class_from_source_map[source_class_id]
#
#     def get_source_class_id(self, class_id, source):
#         """Map an internal class ID to the corresponding class ID in the source dataset."""
#         print('get_source_class_id ..............')
#         info = self.class_info[class_id]
#         assert info['source'] == source
#         return info['id']
#
#     def append_data(self, class_info, image_info):
#         print('append_data ..............')
#         self.external_to_class_id = {}
#         for i, c in enumerate(self.class_info):
#             for ds, id in c["map"]:
#                 self.external_to_class_id[ds + str(id)] = i
#
#         # Map external image IDs to internal ones.
#         self.external_to_image_id = {}
#         for i, info in enumerate(self.image_info):
#             self.external_to_image_id[info["ds"] + str(info["id"])] = i
#
#     @property
#     def image_ids(self):
#         return self._image_ids
#
#     def source_image_link(self, image_id):
#         """Returns the path or URL to the image.
#         Override this to return a URL to the image if it's availble online for easy
#         debugging.
#         """
#         print('source_image_link ..............')
#         return self.image_info[image_id]["path"]
#
#     def load_image(self, image_id):
#         """Load the specified image and return a [H,W,3] Numpy array.
#         """
#         print('load_image ..............')
#         # Load image
#         image = skimage.io.imread(self.image_info[image_id]['path'])
#         # If grayscale. Convert to RGB for consistency.
#         if image.ndim != 3:
#             image = skimage.color.gray2rgb(image)
#         # If has an alpha channel, remove it for consistency
#         if image.shape[-1] == 4:
#             image = image[..., :3]
#         return image
#
#     def load_mask(self, image_id):
#
#         """Load instance masks for the given image.
#
#         Different datasets use different ways to store masks. Override this
#         method to load instance masks and return them in the form of am
#         array of binary masks of shape [height, width, instances].
#
#         Returns:
#             masks: A bool array of shape [height, width, instance count] with
#                 a binary mask per instance.
#             class_ids: a 1D array of class IDs of the instance masks.
#         """
#         # Override this function to load a mask from your dataset.
#         # Otherwise, it returns an empty mask.
#         print ('load_mask ..............')
#         mask = np.empty([0, 0, 0])
#         class_ids = np.empty([0], np.int32)
#         return mask, class_ids
#


   
##########################   DEBUGGER
   
def debug_resize_image():
    image_resized, image_window, scale, padding = resize_image(
            image=np.random.random((100, 200, 3)), min_dim=800, max_dim=1024, min_scale=0,
                            mode='square')
    print (image_resized.shape)
    
def debug_gen_anchors():
    from MaskRCNN_loop.config import config as conf
    resnet_stage_shapes = get_resnet_stage_shapes(conf, image_shape=[1024,1024,3])
    # print(resnet_stage_shapes)

    anchors = gen_anchors(image_shape = [1024,1024,3],
                          batch_size=2, scales=conf.RPN_ANCHOR_SCALES,
                          ratios=conf.RPN_ANCHOR_RATIOS,
                          feature_map_shapes=resnet_stage_shapes,
                          feature_map_strides=conf.RESNET_STRIDES,
                          anchor_strides=conf.RPN_ANCHOR_STRIDE)
    print (anchors.shape, anchors)
    
    
# debug_gen_anchors()



