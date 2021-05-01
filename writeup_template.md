**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

- The code for camera calibration is in `src/camera_calibration.py`.
- Given several chessboard images, their corners' image coordinates `img_points` are found  and world coordinates `obj_points` are defined; then use this pair of coordinates to calculate **camera matrix** and **distort coefficients**; finally use these parameters to undistort images taken by the same camera. ![](output_images/undistorted_image1.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

- Undistort road images taken from the same camera as above: ![](output_images/undistorted_image2.png)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

- the code is in `threshold()` function in `src/pipeline.py`.
- I transferred BGR image to HLS color space and use two HLS channel range to find yellow and white pixels. I referred to this blog https://naokishibuya.medium.com/finding-lane-lines-on-the-road-30cf016a1165.
- The results of (a) undistorted original image, (b) thresholded binary image, (c) perspective transformed binary image with fitted polynomial plotted, and (d) final image with ego lane warped back and radius, car offset displayed.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 585, 460  |   320, 0    |
| 203, 720  |  320, 720   |
| 1127, 720 |  960, 720   |
| 695, 460  |   960, 0    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

---

### Pipeline results for test images

- results for `test1.jpg`

![](output_images/test1_output.png) 

- results for `test2.jpg`

![](output_images/test2_output.png)

- results for `test3.jpg`

![](output_images/test3_output.png)

- results for `test4.jpg`

![](output_images/test4_output.png)

- results for `test5.jpg`

![](output_images/test5_output.png)

- results for `test6.jpg`

![](output_images/test6_output.png)

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

- The final video output is [here](output_images/project_video.mp4).

---

### Discussion

#### 1. problems / issues I faced in implementation.  

- The biggest problem that I faced is the threshold binary image. Although the yellow and white mask worked here, actually it failed on challenge video.

#### 2. Where will the pipeline likely fail?  

- When applied to image with big shadow (for example, under a bridge), white pixel will be hardly detected, because white pixel is taken based on lightness only.

#### 3. to make it more robust?

- I am going to do lane line tracking and sanity check. Right now, since current pipeline can deal with the project video, I skipped them here.