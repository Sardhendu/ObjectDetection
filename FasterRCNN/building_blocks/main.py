#
# import logging
# import tensorflow as tf
# from FasterRCNN.building_blocks import vgg, rpn, proposals
#
# logging.basicConfig(level=logging.DEBUG, filename="logfile.log", filemode="w",
#                     format="%(asctime)-15s %(levelname)-8s %(message)s")
#
# def main():
#     model_path = '/Users/sam/All-Program/App-DataSet/ObjectDetection/FasterRCNN/vgg16_weights.npz'
#     obj_vgg = vgg.vgg16(mode='test', model_path=model_path)
#     input_image, feature_map = obj_vgg.get_feature_map([224, 224, 3])
#
#     # RPN
#     obj_rpn = rpn.rpn(mode='train', feature_map=feature_map)
#     rpn_box_class_prob = obj_rpn.get_rpn_box_class_prob()
#     rpn_bbox = obj_rpn.get_rpn_bbox()
#
#     # PROPOSAL
#     proposals_ = tf.py_func(proposals.get_proposal_wrapper, ['test', rpn_box_class_prob, rpn_bbox], [tf.float32])
#     print (proposals_)
#
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run([proposals_], feed_dict={input_image:})
#
#
#
#
# main()