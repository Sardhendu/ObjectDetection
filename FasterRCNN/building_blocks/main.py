
import logging
import tensorflow as tf
from FasterRCNN.building_blocks import vgg, rpn

logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

def main():
    model_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/FasterRCNN/vgg16_weights.npz'
    obj_vgg = vgg.vgg16(mode='test', model_path=model_path)
    feature_map = obj_vgg.get_feature_map([224, 224, 3])
    
    # print(feature_map.shape)
    #
    # obj_rpn = rpn.rpn(mode='train', feature_map=feature_map)
    # rpn_class_score = obj_rpn.get_rpn_class_sores()
    # rpn_bbox = obj_rpn.get_rpn_bbox()
    #
    # print (rpn_class_score.shape, rpn_bbox.shape)
    
    
    
main()