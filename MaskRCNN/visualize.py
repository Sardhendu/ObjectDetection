

import random
import colorsys
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon



class Visualize():
    def __init__(self, image_path, rows=1, cols=1, figsize=(16, 16)):
        self.image_path = image_path
        self.image = ndimage.imread(image_path, mode='RGB')
        
        
        _, self.axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, facecolor='y', edgecolor='k')
        if cols > 1:
            self.axs = self.axs.ravel()
            
        self.auto_show = True
    
    def gen_random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def vizualize_image(self, imageArray, title_arr=[], data_type='unit8'):
        
        for no, image in enumerate(imageArray):
            self.axs[no].imshow(np.array(image.reshape(image.shape[0], image.shape[1]), dtype=data_type))
            
        if self.auto_show:
            plt.show()
    
    def visualize_boxes(self, boxes, class_ids=None, class_names=None,
                          scores=None, title="",
                          show_mask=True, show_bbox=True,
                          colors=None, captions=None):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """
        # Number of instances
        N = boxes.shape[0]
        # if not N:
        #     print("\n*** No instances to display *** \n")
        # else:
        #     assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        
        # If no axis is passed, create one and automatically call show()
        auto_show = False
        
        
        # Generate random colors
        colors = colors or self.gen_random_colors(N)
        
        # Show area outside image boundaries.
        height, width = self.image.shape[:2]
        self.axs.set_ylim(height + 10, -10)
        self.axs.set_xlim(-10, width + 10)
        self.axs.axis('off')
        self.axs.set_title(title)
        
        masked_image = self.image.astype(np.uint32).copy()
        
        # Loop for each box/proposals
        for i in range(N):
            color = colors[i]
            
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                      alpha=0.7, linestyle="dashed",
                                      edgecolor=color, facecolor='none')
                self.axs.add_patch(p)
            
            # # Label
            # if not captions:
            #     class_id = class_ids[i]
            #     score = scores[i] if scores is not None else None
            #     label = class_names[class_id]
            #     x = random.randint(x1, (x1 + x2) // 2)
            #     caption = "{} {:.3f}".format(label, score) if score else label
            # else:
            #     caption = captions[i]
            # ax.text(x1, y1 + 8, caption,
            #         color='w', size=11, backgroundcolor="none")
            #
            # # Mask
            # mask = masks[:, :, i]
            # if show_mask:
            #     masked_image = apply_mask(masked_image, mask, color)
            #
            # # Mask Polygon
            # # Pad to ensure proper polygons for masks that touch image edges.
            # padded_mask = np.zeros(
            #         (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            # padded_mask[1:-1, 1:-1] = mask
            # contours = find_contours(padded_mask, 0.5)
            # for verts in contours:
            #     # Subtract the padding and flip (y, x) to (x, y)
            #     verts = np.fliplr(verts) - 1
            #     p = Polygon(verts, facecolor="none", edgecolor=color)
            #     ax.add_patch(p)
                self.axs.imshow(masked_image.astype(np.uint8))
        if auto_show:
            plt.show()
