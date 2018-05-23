
'''
The RCNN module takes input as a certain format. For instance the

'''

from skimage.transform import resize
import numpy as np
from MaskRCNN.config import config as conf
import tensorflow as tf


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


def normalize_image(images, mean_pixels):
    """
    Expects an RGB image (or array of images) and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - mean_pixels


def resize_image(image, min_dim, max_dim, min_scale, mode):
    '''
    :param image:       image input
    :param min_dim:     min_dimension as mentioned in config file probably 800
    :param max_dim:     max_dimension as mentioned in config file probably 1024
    :param min_scale:   The scaling factor of height width
    :param mode:        Square matrix or pad64 etc
    :return:
    
    Intuitively, If the image is small say h=600 w=800, but the permissible min_dim=800 and max_dim=1024. We find a scaling factor to convert the larger dimensio (width) to size 1024 and maintain a scaling factor. Using the same scaling factor we also increase the dimension of height.
    
    If the height (smaller dimension is < 1024(min_dim) then we zero pad the image to make it a square matrix)
    '''
    print(image.shape)
    image_dtype = image.dtype

    h, w = image.shape[:2]
    scale = 1

    # When the image size is smaller than the min size permissible in the config file
    # In such a case, we set a scaling factor
    if min_dim:
        scale = max(1, min_dim / min(h, w))
    print (scale)
    
    # If the scale is smaller the permissible scaling in the config file, then we choose the one defined in the config file.
    if min_scale and scale < min_scale:
        scale = min_scale
    print(scale)
    
    # Hmm, Check if your image exceeds the maximum dimension permissible in the config file.
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    print(scale)
    
    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(
                image, (round(h * scale), round(w * scale)),
                order=1, mode="constant", preserve_range=True)
        
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        img_window = (top_pad, h + top_pad, left_pad, w + left_pad)
        print(image.shape)
    else:
        raise ValueError('Code for other mode too.')
    return image.astype(image_dtype), img_window, scale, padding


def process_images(list_of_images, list_of_image_ids):
    '''
    :param list_of_images:
    :return: Three matrices
        1. resized_images = [num_images, h, w, 3]
        2. image_metas = details about image such as
            [image_id, orig_img_h, orig_img_w, orig_img_channel,
             transformed_img_h, transformed_img_w, transformed_img_channel,
             img_window_hTop, img_window_hBottom, img_window_wLeft, img_window_wRight,
             scale_factor, active_class_ids]
        3. image_windows = [img_window_hTop, img_window_hBottom, img_window_wLeft, img_window_wRight,]
    '''
    transformed_images = []
    image_metas = []
    img_windows = []
    for img, img_id in zip(list_of_images, list_of_image_ids):
        transformed_image, img_window, scale, padding = resize_image(img, conf.IMAGE_MIN_DIM, conf.IMAGE_MAX_DIM, conf.IMAGE_MIN_SCALE, conf.IMAGE_RESIZE_MODE)

        transformed_image = normalize_image(transformed_image, conf.MEAN_PIXEL)

        # COmpose a metadata of images for debugging and
        image_meta = compose_image_meta(
                img_id, img.shape, transformed_image.shape, img_window, scale,
                np.zeros([conf.NUM_CLASSES], dtype=np.int32))

        transformed_images.append(transformed_image)
        image_metas.append(image_meta)
        img_windows.append(img_window)

    transformed_images = np.stack(transformed_images)
    image_metas = np.stack(image_metas)
    img_windows = np.stack(img_windows)
    return transformed_images, image_metas, img_windows
        
# image_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/images/3627527276_6fe8cd9bfe_z.jpg'
#
# input_image_to_tensor([image_path])