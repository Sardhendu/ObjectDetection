

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
        
3. Output 2: (Regressor to refine the bounding box): In a regressor we would like to predict 32 (9x4) dimensional depth vector where 9 is the number of anchors and 4 is the bounding box representation. From the Output 1, we already have the anchors that are foreground and background. In a regressor we don't use the background anchors because we dont have ground truth boxes for them.
    
    * Here we use a smooth L1 loss on the position (x,y) of top left of the box and logarithm of the heights and width.
    
    
## ROI Pooling (Detecting Objects in the boxes):




## MASK RCNN
The Mask RCNN has a different taste when it comes down to detecting feature map and object. It uses the FPN (Feature Pyramid Net) to detect feature maps which in-turn are used by RPN (Region Proposal Network) to detect object. Note: There's nothing stopping us to use FPN with RPN in Fast-RCNN or Faster RCNN. 

Extra Notes:
FPN: https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
    
    

 
              
     