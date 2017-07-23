**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/carnotcar.png
[image2]: ./writeup_images/HOG_example.jpg
[image3]: ./writeup_images/sliding_windows.jpg
[image4a]: ./writeup_images/output_4_0.png
[image4b]: ./writeup_images/output_4_1.png
[image4c]: ./writeup_images/output_4_2.png
[image4d]: ./writeup_images/output_4_3.png
[image4e]: ./writeup_images/output_4_4.png
[image5a]: ./writeup_images/output_5_0.png
[image5b]: ./writeup_images/output_5_1.png
[image5c]: ./writeup_images/output_5_2.png
[image5d]: ./writeup_images/output_5_3.png
[image5e]: ./writeup_images/output_5_4.png
[image6a]: ./writeup_images/output_7_0.png
[image6b]: ./writeup_images/output_7_1.png
[image6c]: ./writeup_images/output_7_2.png
[image6d]: ./writeup_images/output_7_3.png
[image6e]: ./writeup_images/output_7_4.png
[image7]: ./writeup_images/detection.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook

I started by reading in all the `vehicle` and `non-vehicle` images and explored some statistical information about the dataset in `data_exploration.ipyhb` notebook. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

####2. Explain how you settled on your final choice of HOG parameters.
### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12  # HOG orientations
pix_per_cell = 10 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

I tried various combinations of parameters for color space, spatial binning, color histogram and HOG features. I know that RGB color space is unstable to illumination changes therefore I decided to use `YCrCb` color space. As HOG parameters I used, 12 different orientations, 10 pixels per cells and 2 cells per block. In the beginning, I only extracted HOG features from the first channel, luminance, of the image. However, I found out that using 3 channels improves the performance which in turn increases processing time per image a lot. I used 16 bins for color histogram and 16x16 spatial binning. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all the images from car and not car dataset. I only used GIT and KITTI dataset images. I decided to use all the feature types in the training process including color histogram, spatial binning and HOG features. After concatenating all the features, I used `StandardScaler()` to scale different types of features to similar ranges. Then, used Linear SVM to train car/notcar classifier. SVM training took just ~7 seconds which is way faster than using Neural Networks!!! That's why I like traditional CV methods =) and I got 0.993 accuracy. Of course, this is a very simple one class classification problem.

```
Using: 12 orientations 10 pixels per cell and 2 cells per block
Feature vector length: 4416
7.46 Seconds to train SVC...
Test Accuracy of SVC =  0.993
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First, I decided on 4 different scales to cover all the appearances of the cars at different distances. I used sliding windows of sizes 64x64, 128x128, 196x196 and 256x256. Then, slided windows in the lower half of the image with 0.75 overlap. Because per frame processing time increases linearly with the total number of slidipng windows, I performed sliding window search only on the right upper half of the image as the project video contains car on the right of the egocar only. Here is the different scales of sliding windows that I used:

![alt text][image4a]
![alt text][image4b]
![alt text][image4c]
![alt text][image4d]
![alt text][image4e]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 

Here are some detections:

![alt text][image5a]
![alt text][image5b]
![alt text][image5c]
![alt text][image5d]
![alt text][image5e]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/oclIsCA3By0)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used heatmap approach suggested in the tutorials. For this, first, I extracted car candidates in the form of bounding box, then I stored bounding boxes from each frame in a fixed size queue structure implemented in `vehicle_detection()` class. When detection result is queried, I simply sum all the inner areas of detection windows inside the queue and represent them as heatmap. I used 25 for both the queue size and threshold level. In order to eliminate false detections, I threshold resulting heatmap and use `label()` function to label uniquely each connected components in the thresholded heatmap. Then, I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6a]
![alt text][image6b]
![alt text][image6c]
![alt text][image6d]
![alt text][image6e]

### Here is the result after thresholding heatmap image, labeling and encompassing the labeled regions with bounding boxes:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. First of all, processing time is an issue. However there are many points to optimize to decrease processing time per frame. Currently, HOG features are calculated for each sliding window but most of the sliding windows are overlapping with each other. Therefore, we can calculate HOG features once for an image and crop it accordingly.

2. Number and scale of sliding windows affect detection results. It is possible to get your detections better fit on vehicle if you increase the number of sliding windows. But this increases the processing time as well.

3. Heatmap approach is simple and works well in average. However, it's is hard to decide heatmap threshold parameter and it definitely affects the shape of the bounding box fitted on the vehicle. Morphological operations on the heatmap may produce better results.
