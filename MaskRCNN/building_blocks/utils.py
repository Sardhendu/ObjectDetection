


import tensorflow as tf
import numpy as np


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

def generate_anchors_for_feature_map(scales, ratios, feature_map_shape, feature_map_stride, anchor_strides):
    """
    Generates anchors for each feature map
    
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
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
        Example: If feature_map_shape = [64x64], scale = 128 and feature_map_stride = 16
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
          --              (-45,90)---------
          
          Using Step 3 and Step 4 we output a [12288 x 4] dimensional output.
        
    
         
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    #
    # # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    #
    # # Enumerate shifts in feature space
    shifts_y = np.arange(0, feature_map_shape[0], anchor_strides) * feature_map_stride
    shifts_x = np.arange(0, feature_map_shape[1], anchor_strides) * feature_map_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])


    # # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    return boxes


def gen_anchor(scales, ratios, feature_shapes, feature_strides, anchor_strides):
    for i in range(0,len(scales)):
        if feature_shapes[i][0] == 64:
            print ('running for ', scales[i], feature_shapes[i], feature_strides[i])
            generate_anchors_for_feature_map(scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_strides)
        # break
        

def debugg():
    from MaskRCNN.config import config as conf
    resnet_stage_shapes = get_resnet_stage_shapes(conf, image_shape=[1024,1024,3])
    print(resnet_stage_shapes)

    gen_anchor(scales=conf.RPN_ANCHOR_SCALES, ratios=conf.RPN_ANCHOR_RATIOS,
               feature_shapes=resnet_stage_shapes, feature_strides=conf.RESNET_STRIDES,
               anchor_strides=conf.RPN_ANCHOR_STRIDE)
    
debugg()
    