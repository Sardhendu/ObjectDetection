

Source: https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8

1. The Faster RCNN contains two network
    * RPN for proposing region
    * Network to detect object using the proposal from RPN
  
## RPN:  
    
1. Anchor: Internally RPN ranks region boxes called Anchors and proposes the one likely to contain the object
    * At any pixel point the Faster-RCNN default configuration defines 9 different anchors. Each of these anchols have different dimension of bounding box and different scale. Example one anchor may take 128x128 in dimension while the other may take 256x256. They can be changed depending the domain.
    
    Let say 
    * We have a 600x800 image,
    * We choose a pixel (one position for anchor selection) after every 16 stride, 
    * Say, we have total of 9 anchors.
    
    Then position in x axis = ceil(600/16 + 1) = 39
    Then position in y axis = ceil(800/16 + 1) = 51
    
    Which means total number of boxes to consider is 39*51*9 = 17901. Which is very high, therefore these these methods are very good at region proposal and bounding boxes. Despite a anchor search of 17901 these methods are very fast compared to selective search becasue here they are operated through RPN i.e Region proposal network where their is a lot of parameter sharing across the different network.
    
2. Output 1: (Foreground-Background Classifier) The output of the RPN is a bunch of boxes that will be evaluated or classified by the classifier and regressor to check the occurance of the object. RPN performs a binary classification, i.e. predicts if an anchor is foreground and background.
    
    * Get labels to anchors:
        * Input to the (Classifier): Ground truth and bunch of anchors received from previous layers
        * Method: 
            * Label the anchors as Foreground that have high intersection with the ground truth
            * Label the anchors as Background if the intersection is low.
    
    * Get features for anchors: (A fully convolutional way) Now that we know which anchor is foreground and which anchor is background.
        * Lets say that we perform convolution and downsampling on a 600x800 image for 16 iteration and shrink it to (38x51xDepth) feature map, also let suppose we have 9 anchors per pixel of the feature map, then all we have to do is make another convolution to convert the output map to (38x51x18), where 18 represent binary outcome to each anchor at every pixel position of the feature map.
        
    * Receptive field:
        
3. Output 2: (Regressor to refine the bounding box): In a regressor we would like to predict 36 (9x4) dimensional depth vector where 9 is the number of anchors and 4 is the bounding box representation. From the Output 1, we already have the anchors that are foreground and background. In a regressor we don't use the background anchors because we dont have ground truth boxes for them.
    
    * Here we use a smooth L1 loss on the position (x,y) of top left of the box and logarithm of the heights and width.
    
    
## ROI Pooling (Detecting Objects in the boxes):




## MASK RCNN
The Mask RCNN has a different taste when it comes down to detecting feature map and object. It uses the FPN (Feature Pyramid Net) to detect feature maps which in-turn are used by RPN (Region Proposal Network) to detect object. Note: There's nothing stopping us to use FPN with RPN in Fast-RCNN or Faster RCNN. 

Extra Notes:
FPN: https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
    
    

 
 
 
 
#### ABOUT THE WHOLE MODULE in short:
1. Preprocess (Preprocessing the image to be feed into the network):
    1. Resize the image with proper scale (aspect ratio):
        * Use a scale to resize (resize done using bilinear interpolation or normal resize) 
        * If the image < [1024, 1024] then perform bilinear interpolation to resize image to [1024, 1024]
        * If the image > [1024, 1024] then perform normal resize operation to form [1024, 1024]
        * In case if the image_height != image_weight perform zero padding to the smaller dimension
        
    2. Normalize the image with pretrained means
    
    3. Generate anchors (Note anchors are in pixel scale)

2. FPN(Feature pyramid network): This step is basically used to generate feature maps at different scales.
    We generate 4 different scale feature maps each with a depth of 256
        *   P2=[num_batches, 256, 256, 256], total pixels
        *   P3=[num_batches, 128, 128, 256], 
        *   P4=[num_batches, 64, 64, 256], 
        *   P5=[num_batches, 32, 32, 256]
        
3. RPN (Region Proposal Networks): Performs two steps.
    Assumption:
        * anchor_stride=1 and anchor_per_pixel = 3, this means that at every pixel in the feature map we generate 3 anchors. 
        
    1. Generate rpn_class_prob: says if the pixel is foreground or background 
        * For feature map 256x256 (P2): [num_batch, 256x256x3, 2], 2-> (bg_prob, fg_prob)
        * For feature map 256x256 (P3): [num_batch, 128x128x3, 2], 2-> (bg_prob, fg_prob)
        * For feature map 256x256 (P4): [num_batch, 64x64x3, 2], 2-> (bg_prob, fg_prob)
        * For feature map 256x256 (P5): [num_batch, 32x32x3, 2], 2-> (bg_prob, fg_prob)
        
    2. Generate bounding boxes: given that the pixel is foreground, this module will generate all the anchors at each pixel position.
        * For feature map 256x256 (P2): [num_batch, 256x256x3, 4], 4-> (dy, dx, log(dh), log(dw))
        * For feature map 256x256 (P3): [num_batch, 128x128x3, 4], 3-> (dy, dx, log(dh), log(dw))
        * For feature map 256x256 (P4): [num_batch, 64x64x3, 4], 3-> (dy, dx, log(dh), log(dw))
        * For feature map 256x256 (P5): [num_batch, 32x32x3, 4], 3-> (dy, dx, log(dh), log(dw))
        
3. Proposals ()
    Required 3 inputs
    1. rpn_class_probs
    2. rpn_bbox
    3. input_anchors: [num_batches, num_anchors, 4]
        * The anchor generation depends on how many feature maps are produces by FPN and the anchor strides.
        * Say, if the feature_map is [256x256] and anchor_stride = 1, If anchors per pixel position is 3 then the total number of anchors = 256x256c3 = 196608.
        * If using an FPN model then we calculate the anchors for each feature map and append anchors generated from each feature map.
        * Normalized Anchors:
            * We noralize the anchors in the scale od 0-1, since we make the bounding box prediction in nornalized scale
            * Say image_shape = [1024,1024,3] and we have an anchor (box) at coordnates [212,212,712,712] then we normalize it to [212/1023, 212/1023, (712-1)/1023, (712)/1023]
        
    Output:
    1. proposals
              
     