


import numpy as np
from skimage import transform
import logging

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


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
    image_window = (top_pad, left_pad, h + top_pad, w + bottom_pad)
    
    return image.astype(image.dtype), image_window, scale, padding

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


def norm_boxes(anchors, shape):
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
    shift = np.array([0, 0, 1, 1])
    return np.divide((anchors - shift), scale).astype(np.float32)

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
        anchors.append(generate_anchors_for_feature_map(scales[i], ratios, feature_map_shapes[i], feature_map_strides[i],
                                                     anchor_strides))
    anchors = np.concatenate(anchors, axis=0)
    logging.info('Anchors: concatenated for each stage: shape = %s', str(anchors.shape))
    anchors = np.broadcast_to(anchors, (batch_size,) + anchors.shape)
    logging.info('Anchors: Broadcast to num_batches: shape = %s', str(anchors.shape))

    # Normalize Anchors
    anchors = norm_boxes(anchors, shape=image_shape[:2])
    return anchors
    
    
    



   
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



