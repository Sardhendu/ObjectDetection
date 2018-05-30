



Faster RCNN:


When Testing:

1. Consists of three stages
    -> A Conv net to generate feature maps. Here we use VGG-16
    -> A RPN module that uses the feature map outputs to generate the bounding box.
        Includes
        --> Classification part
        --> Regression part
    --> A Proposal Layer: Find Anchor proposals