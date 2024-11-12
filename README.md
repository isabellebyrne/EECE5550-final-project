## EECE5550 Project Proposal

**Team Members**
- Anirudh Muthuswamy
- Isabelle Byrne
- Charles Amente
- Hebron Taylor


**Motivation**
A key component of any SLAM algorithm is its ability to accurately recognize when it has revisited a mapped location. The typical methodology involves combining a front-end feature detector/extractor and a feature descriptor generator, along with a back-end loop closure, to increase map accuracy and improve localization.

Traditional front-end methods entail extracting hand-crafted features, which are not robust as the environment and scenarios vary. Advances in deep learning, particularly CNNs, make it possible to extract learned, more adaptive features. However, CNNs tend to be bulky and consume many computational resources.

**Proposal** 
*For this project we would like to explore the idea of creating a lightweight CNN targeted towards resource constrained platforms. The lightweight CNN should retain if not bolster loop closure performance. In short, our plan is to augment front-end SLAM feature detection/extraction and feature descriptor generation, as well as possibly improve the back-end loop closure algorithm (if we have time).*

In order to create a lightweight CNN we first plan on using non-lightweight CNN and then using techniques such as pruning, quantization, architecture design in order to compress our model 

To benchmark our work we plan on comparing our lightweight CNN approach to traditional feature extractor/descriptor methods such as SURF, SIFT and ORB. These comparisons will enable us to determine if the quality of features extracted and the descriptors generated from the lightweight CNN are better or worse than traditional methods.


The project can be broken down into the following phases:

1. Get CNN based descriptor/extractor setup
2. Get traditional descriptor/extractor setup
3. Compare with traditional SLAM descriptors/extractor
4. CNN based descriptor/extractor setup for improvements (meant for resource constrained robotics systems so the improvements would be CPU usage and memory footprint improvements)
5. Similarity score improvements
6. Integrate with ORB-SLAM (maybe; time permitting)


**Problems being addressed**

Traditional loop closure detection algorithms use Bag-of-Visual-Words (BoVW) models with hand-crafted features, such as SIFT, SURF, ORB, and BRIEF. This type of approach has several limitations in terms of robustness and accuracy across different environments. Recent advances in Deep Learning, particularly with CNNs, have led to better performance when it comes to image representation tasks due to their ability to exploit learned features instead of hand-crafted traditional features. 

We also plan on creating a CNN for the purpose of using it on resource constrained platforms, allowing cost-effective and accessible implementations.

**Current Plan of Implementation:**

Image Representation
: Given an Input Image, we extract feature vectors from the l-th layer of the CNN model, where d is the number of dimensions in the fully connected layer.  The feature vectors are then normalized using L2 normalization. 

Dimensionality Reduction
: We would apply PCA to reduce the dimensionality of the feature vectors from d to dreduced 

Whitening 
: The reduced feature vectors are then whitened to decorrelate features and normalize their variances. This helps to focus on fundamental patterns in data and removing redundant information. 

Similarity Score 
: Finally, the key step in visual loop closure detection is the estimation of similarity between frames, and in our case we would calculate the distance between CNN feature vectors of different frames using a similarity score (currently with Euclidean distance metric).

**Experiments planned**
Since this project aims at exploring the difference between traditional BoVW models with hand-crafted features and machine learning based methods, our experiments will revolve around comparing feature descriptor complexity and accuracy between traditional and ML based approaches. Specifically this would look like varying the descriptor length as well as descriptor data type and seeing how well the methods are able to detect image features once the image has been modified.

To empirically quantify our results we will look at the following:
1. Image recognition rate
2. CPU time consumed for each method
3. Memory consumption for each method


To test image recognition rate we will first choose a set of ground truth test images. Next we will augment these images such that they account for:
- Compression artifacts
- Viewpoint changes
- Illumination changes
- Image blur

Once this is done we can compare the ground truth and perturbed image for each approach to get an image recognition rate.

These images will be pulled from AUT-VI dataset

**Project Deliverables**
The output of this project will be a set of code that enables users:
1. Use a CNN approach to generate a feature descriptor
2. Use a traditional method (SURF, SIFT, BRIEF) to generate a feature descriptor
3. Run a script to compare the ML and traditional methods
4. A script that integrates the CNN approach into a ORB-SLAM as its front-end (this is a stretch goal)

All code will be uploaded to a publically available github repository.


Papers & Resources
[1]  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8082072

[2] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10062206

[3] https://arxiv.org/pdf/1504.05241

[4] https://github.com/craymichael/CNN_LCD

[5] https://a3dv.github.io/autvi
