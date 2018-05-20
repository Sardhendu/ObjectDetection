
import os
import json
import xml.etree.ElementTree as ET

def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}

    with open(ann_dir) as f:
        dat = json.load(f)
    print(type(dat))
    for num, (key, val) in enumerate(dat.items()):
        print (num)
        print (key)
        print (val)
        print('')
        
        if num == 2:
            break
        
    #     for elem in tree.iter():
    #         if 'filename' in elem.tag:
    #             img['filename'] = img_dir + elem.text
    #         if 'width' in elem.tag:
    #             img['width'] = int(elem.text)
    #         if 'height' in elem.tag:
    #             img['height'] = int(elem.text)
    #         if 'object' in elem.tag or 'part' in elem.tag:
    #             obj = {}
    #
    #             for attr in list(elem):
    #                 if 'name' in attr.tag:
    #                     obj['name'] = attr.text
    #
    #                     if obj['name'] in seen_labels:
    #                         seen_labels[obj['name']] += 1
    #                     else:
    #                         seen_labels[obj['name']] = 1
    #
    #                     if len(labels) > 0 and obj['name'] not in labels:
    #                         break
    #                     else:
    #                         img['object'] += [obj]
    #
    #                 if 'bndbox' in attr.tag:
    #                     for dim in list(attr):
    #                         if 'xmin' in dim.tag:
    #                             obj['xmin'] = int(round(float(dim.text)))
    #                         if 'ymin' in dim.tag:
    #                             obj['ymin'] = int(round(float(dim.text)))
    #                         if 'xmax' in dim.tag:
    #                             obj['xmax'] = int(round(float(dim.text)))
    #                         if 'ymax' in dim.tag:
    #                             obj['ymax'] = int(round(float(dim.text)))
    #
    #     if len(img['object']) > 0:
    #         all_imgs += [img]
    #
    # return all_imgs, seen_labels


ann_dir = '/Users/sam/All-Program/App-DataSet/z-others/annotations/person_keypoints_val2014.json'
parse_annotation(ann_dir=ann_dir, img_dir=None, labels=[])