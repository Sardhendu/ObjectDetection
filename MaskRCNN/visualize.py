

import random
import colorsys
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")
    
    
class Visualize():
    def __init__(self, image_path=None, rows=1, cols=1, figsize=(5, 5)):
        if image_path:
            self.image_path = image_path
            self.image = ndimage.imread(image_path, mode='RGB')
        
        
        _, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, facecolor='y', edgecolor='k')
        if cols > 1:
            self.axs = axs.ravel()
        else:
            self.axs = [axs]
            
        self.auto_show = True
        self.num = 0
    
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
    
    def vizualize_image(self, image=None, data_type='uint8'):
        if image is not None:
            self.image = image
            
        self.axs[self.num].imshow(np.array(self.image, dtype=data_type))
        self.num += 1

    def visualize_image_2d(self, imageArray, title_arr=[], data_type='uint8'):
        for image in imageArray:
            self.axs[self.num].imshow(np.array(image.reshape(image.shape[0], image.shape[1]), dtype=data_type))
            self.num += 1
    
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
        self.axs[self.num].set_ylim(height + 10, -10)
        self.axs[self.num].set_xlim(-10, width + 10)
        self.axs[self.num].axis('off')
        self.axs[self.num].set_title(title)
        
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
                self.axs[self.num].add_patch(p)
            
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
                self.axs[self.num].imshow(masked_image.astype(np.uint8))
    
    def show(self):
        plt.show()
