# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./images/hog.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./images/hog1.png
[image9]: ./images/hog2.png
[image10]: ./images/hog3.png
[image11]: ./images/yuv.png
[image12]: ./images/detect1.png
[image13]: ./images/detect2.png
[image14]: ./images/heat1.png
[image15]: ./images/heat2.png
[image16]: ./images/final.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook `VehicleDetect.ipynb`. The parameters are contained in the fourth code cell.

I started by reading in all the `vehicle` and `non-vehicle` images. And started the classifier with HOG features only initially.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Initially started with RGB and 12 orientations, 12 pixels per cell. That gave less number of detections so I finally ended up with 9 orientations and 8 pixels per cell. Later I kept all parameters constant and changed the color space and chose that color space which gave highest accuracy and detected all vehicles in the test image

After working for hours(HOG computations is expensive and therefore pretty slow :( ), I've come up with the following parameters
```python
color_space = 'YUV'
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" 
```
Original image and corresponding YUV converted image
![alt text][image11]

The HOG features for the three channels of YUV image above. These features are then trained over SVM
![alt text][image8]
![alt text][image9]
![alt text][image10]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In code cell 5, I've trained the classifier using a Linear SVM. I've tried DecisionTreeClassifier as well as SVM with rbf kernel but they are over-fitting so I finally ended up using LinearSVM

I've used HOG, spatial and color features with the following parameters.  
```python
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
```
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I only searched between y = 400 to y = 700. And I've used 70% overlap with two sized windows 64 x 64 and 32 x 32. 

In codecell 5, I have a function draw() which loads all the windows and scans them for every image from the video

The decision of using 70% overlap is based on the fact that it gave better results when compared to 50%, most probably because of the heatmap algorithm which I applied at later stages with threshold 1.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image12]
![alt text][image13]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=6lg39nTRLCY)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I 
constructed bounding boxes to cover the area of each blob detected. The part above is mostly done as explained in the video lectures.

**The part where I've made siginicant changes are in the video pipeline to remove false positives that spuriously occur**. The algorithm is as follows

```python
# bboxes are list of all distinct possible rectangles that can be cars and bbox is the current rectangle from heatmap
for (ind,(box, count)) in enumerate(bboxes):
    mid = [(box[0][0] + box[1][0])/2, (box[0][1] + box[1][1])/2]
    bmid = [(bbox[0][0] + bbox[1][0])/2, (bbox[0][1] + bbox[1][1])/2]
    
    # If the distance between the exisitng box and the current heatmap is less than 300 then we consider it to be a part of the exisiting box
    if abs(mid[0] - bmid[0]) + abs(mid[1] - bmid[1]) < 300:
        if area(bbox) > area(maxArea[ind]):
            maxArea[ind] = bbox
        # increase the weight of this bbox
        bboxes[ind][1] += 1
        break
else:
    maxArea[len(bboxes)] = bbox 
    bboxes.append([bbox,1])

for (ind,(box, count)) in enumerate(bboxes):
    # If the bbox has no heatmap in this frame, decrease its weight
    if area(maxArea[ind]) is 0 and bboxes[ind][1] >= 2.0:
        bboxes[ind][1] -= 2.

for (ind, (bbox, count)) in enumerate(bboxes):
    fbox = maxArea[ind] if area(maxArea[ind]) > area(bbox) else bbox
    if (area(maxArea[ind]) > bboxes[ind][1]) or (area(maxArea[ind]) > 2500):
        bboxes[ind][0] = maxArea[ind]
#         area check for noise reduction if any
    # Draw those boxes which have atleast 15 weight and has an area of atleast 1500 pixels
    if (count > 15) and area(fbox) > 1500:
        cv2.rectangle(img, fbox[0], fbox[1], (255,0,0), 8)
# Return the image
return img
```


Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are heat map frames and their corresponding heatmaps:

![alt text][image14]
![alt text][image15]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image16]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

