import random
import cv2
import math
import numpy as np
from MaskRCNN.building_blocks import utils


class Dataset():
    def __init__(self, num_images, height, width, num_classes):
        self.image_meta = {}
        self.num_classes = num_classes
        
        for i in range(0, num_images):
            self.image_meta[i] = self.build_images_meta(height, width)
    
    def draw_bg_image(self, height, width, bg_color):
        bg_ = np.array(bg_color).reshape([1, 1, 3])
        bg_image = np.ones([height, width, 3]) * np.array(bg_, dtype=np.uint8)
        return bg_image
    
    def draw_object_shape(self, image, object_info):
        ''' WHY THE WEIRDNESS IN FORMULA

        :param bg_image:
        :param object_info:
        :return:

        Important Note: When you look at the formulas, it might seem weird or rather oposite to what we are
        accustomed to use with numpy. This is because, we use OpenCV.

        Numpy          0,10 ------------ 10,10
                            |          |
                            |          |
                            |          |
                       0,0  |__________| 10,0

        OpenCV         0,0  ------------ 10,0
                            |          |
                            |          |
                            |          |
                       0,10 |__________| 10,10

        '''
        object_, (color), (c_y, c_x, size) = object_info
        if object_ == 'square':
            cv2.rectangle(image, (c_x - size, c_y - size), (c_x + size, c_y + size), color, -1)
        elif object_ == 'circle':
            cv2.circle(img=image, center=(c_x, c_y), radius=size, color=color, thickness=-1)
        elif object_ == 'triangle':
            points = np.array([[(c_x, c_y - size),  # Top point
                                (c_x - size / math.sin(math.radians(60)), c_y + size),  # Bottom left
                                (c_x + size / math.sin(math.radians(60)), c_y + size),  # Bottom right
                                ]], dtype=np.int32)
            
            cv2.fillPoly(image, points, color)
        return image
    
    def gen_random_shapes(self, height, width):
        # select a random object (class)
        object_ = np.random.choice(['square', 'triangle', 'circle'])
        
        # Get random color for 3 channels
        color = tuple([random.randint(0, 255) for _ in range(3)])
        
        # Leave a buffer space (pad) of 20 pixels for the object_ to accomodate in the
        # background and collect a random center points (c_x, cy)
        buffer_space = 20
        c_y = np.random.randint(buffer_space, height - buffer_space - 1)
        c_x = np.random.randint(buffer_space, width - buffer_space - 1)
        
        # Get a Random size of the bounding box in which the object_ (tringle, square, cicle) is embedded
        size = np.random.randint(buffer_space, height // 4)
        # to account for both side towards the left form center to the right
        return object_, color, (c_y, c_x, size)
    
    def gen_random_image(self, height, width):
        # Pick a random 3 channel for the background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        
        # Pick randomly how many object_ to put in the background image frame
        num_objects = np.random.randint(1, self.num_classes)
        
        object_info = []
        bounding_boxes = []
        for _ in range(0, num_objects):
            object_, color, (c_y, c_x, size) = self.gen_random_shapes(height, width)
            object_info.append((object_, (color), (c_y, c_x, size)))
            bounding_boxes.append(
                    [c_y - size, c_x - size, c_y + size, c_x + size]
            )  # lower left and upper right coordinates
        bounding_boxes = np.array(bounding_boxes)
        # print(bounding_boxes)
        # Sometimes if we select two or more objects to be dispayed in the image we can have those images
        # to overlap completely. In such a case we should ensure that the non-max supression between the
        # objects are atleast 0.3 so that we dont mess out training labels.
        keep_idx = utils.non_max_supression(bounding_boxes, np.arange(num_objects), threshold=0.3)
        # print('object_info pre NMS ', object_info)
        object_info = [s for i, s in enumerate(object_info) if i in keep_idx]
        # print('keep_idx ', keep_idx)
        # print('objects post NMS ', object_info)
        return bg_color, object_info
    
    def build_images_meta(self, height, width):
        image_info = {}
        image_info['height'] = height
        image_info['width'] = width
        bg_color, object_info = self.gen_random_image(height, width)
        image_info['object_info'] = object_info
        image_info['bg_color'] = bg_color
        return image_info
    
    def get_images(self, image_id):
        image_info = self.image_meta[image_id]
        objects_info = image_info['object_info']
        bg_color = image_info['bg_color']
        height = image_info['height']
        width = image_info['width']
        image = self.draw_bg_image(height, width, bg_color)
        print(objects_info)
        num_objects = len(objects_info)
        
        for i in np.arange(num_objects):
            image = self.draw_object_shape(image, objects_info[i])
        return image


data = Dataset(num_images=2, height=128, width=128, num_classes=4)
# print (data.image_meta)
data.get_images(image_id=0)
# data.build_image_meta(height=128, width=128)
