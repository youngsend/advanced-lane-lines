from src.camera_calibration import *
from src.utility import *


class Pipeline:
    def __init__(self):
        # camera calibration
        self.camera_calibrator = CameraCalibration('camera_cal/')

        # perspective transform
        self.img_shape = (720, 1280, 3)
        self.perspective_src = np.float32(
            [[197, 720],
             [592, 450],
             [688, 450],
             [1118, 720]])
        self.perspective_dst = np.float32(
            [[340, 720],
             [340, 0],
             [940, 0],
             [940, 720]])
        self.perspective_M = cv2.getPerspectiveTransform(src=self.perspective_src, dst=self.perspective_dst)
        self.perspective_Minv = cv2.getPerspectiveTransform(src=self.perspective_dst, dst=self.perspective_src)

        # curve fitting
        self.left_fit = None
        self.right_fit = None
        # self.left_fit = np.array([2.13935315e-04, -3.77507980e-01, 4.76902175e+02])
        # self.right_fit = np.array([4.17622148e-04, -4.93848953e-01, 1.11806170e+03])
        self.margin = 100  # Set the width of the windows +/- margin

        # radius and center offset
        self.ym_per_pix = 35 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 600  # meters per pixel in x dimension

    def process(self, img):
        """pipeline function with whole steps."""
        undistorted = self.camera_calibrator.undistort(img=img)
        mask = self.threshold(undistorted)
        warped_mask = self.warp_img(mask)
        fitted = self.fit_polynomial(warped_mask)
        final = self.visualize_ego_lane(undistorted, self.left_fit, self.right_fit)
        return undistorted, mask, warped_mask, fitted, final

    @staticmethod
    def threshold(img):
        """use hls threshold to get white and yellow pixel mask.
        this function refers to https://github.com/naokishibuya/car-finding-lane-lines
        """
        # when used to process video, RGB2HLS; when used to process cv2 image, BGR2HLS
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # white mask
        lower = np.uint8([0, 200, 0])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(hls, lower, upper)

        # yellow mask
        lower = np.uint8([10, 0, 100])
        upper = np.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(hls, lower, upper)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return mask

    def warp_img(self, img):
        warped = cv2.warpPerspective(src=img, M=self.perspective_M,
                                     dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped

    def warp_img_back(self, warped):
        img = cv2.warpPerspective(src=warped, M=self.perspective_Minv,
                                  dsize=(warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)
        return img

    def find_lane_pixels_sliding_window(self, binary_warped):
        """Use sliding window method to find lane pixels, used when left_fit and right_fit are None."""
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int32(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int32(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & (
                    nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & (
                    nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            # (`right` or `leftx_current`) on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Colors in the left and right lane regions. Here I use BGR...
        out_img[lefty, leftx] = [0, 0, 255]
        out_img[righty, rightx] = [255, 0, 0]

        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, binary_warped):
        # find lane pixels
        if (not self.left_fit is None) and (not self.right_fit is None):
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels_around_poly(
                binary_warped, self.left_fit, self.right_fit)
            out_img = self.visualize_search_area(out_img, self.left_fit, self.right_fit)
        else:
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels_sliding_window(binary_warped)

        # fit polynomial using lane pixels
        self.left_fit, self.right_fit = self.fit_poly(leftx, lefty, rightx, righty)
        left_fitx, right_fitx, ploty = self.get_poly_coordinates(
            binary_warped.shape, self.left_fit, self.right_fit)

        # Plots the left and right polynomials on the lane lines
        left_coordinates = np.stack((np.int32(left_fitx), np.int32(ploty)), axis=-1).reshape((-1, 1, 2))
        out_img = cv2.polylines(out_img, [left_coordinates], False, (0, 255, 255), thickness=5)
        right_coordinates = np.stack((np.int32(right_fitx), np.int32(ploty)), axis=-1).reshape((-1, 1, 2))
        out_img = cv2.polylines(out_img, [right_coordinates], False, (0, 255, 255), thickness=5)

        return out_img

    def fit_poly(self, leftx, lefty, rightx, righty):
        # Fit a second order polynomial to each with np.polyfit()
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit

    def get_poly_coordinates(self, img_shape, left_fit, right_fit):
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        # calc both polynomials using ploty, left_fit and right_fit
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        return left_fitx, right_fitx, ploty

    def find_lane_pixels_around_poly(self, binary_warped, left_fit, right_fit):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set the area of search based on activated x-values
        # within the +/- margin of our polynomial function
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - self.margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                                  left_fit[1] * nonzeroy +
                                                                                  left_fit[2] + self.margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - self.margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                                    right_fit[1] * nonzeroy +
                                                                                    right_fit[2] + self.margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Color in left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        return leftx, lefty, rightx, righty, out_img

    def visualize_search_area(self, out_img, left_fit, right_fit):
        window_img = np.zeros_like(out_img)
        left_fitx, right_fitx, ploty = self.get_poly_coordinates(out_img.shape[:2], left_fit, right_fit)
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return result

    def visualize_ego_lane(self, out_img, left_fit, right_fit):
        left_fitx, right_fitx, ploty = self.get_poly_coordinates(out_img.shape[:2], left_fit, right_fit)
        left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        line_points = np.hstack((left_line_window, right_line_window))

        # draw lane polygon on original image
        window_img = np.zeros_like(out_img)
        cv2.fillPoly(window_img, np.int_([line_points]), (0, 255, 0))
        # warp window_img back to front camera viewpoint
        window_img = self.warp_img_back(window_img)
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # add text of radius and center offset
        left_curverad, right_curverad = self.measure_curvature_real(self.left_fit, self.right_fit)
        # right offset: -, left offset: +.
        center_offset = (self.img_shape[1] - left_fitx[-1] - right_fitx[-1]) * self.xm_per_pix / 2.0
        text1 = 'left lane line radius: {:.4f}m'.format(left_curverad)
        text2 = 'right lane line radius: {:.4f}m'.format(right_curverad)
        text3 = 'horizontal car offset: {:.4f}m'.format(center_offset)
        cv2.putText(result, text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, text2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, text3, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return result

    def measure_curvature_real(self, left_fit, right_fit):
        """Calculates the curvature of polynomial functions in meters."""
        # Define conversions in x and y from pixels space to meters

        xm_to_ym = self.xm_per_pix / self.ym_per_pix

        # convert pixel poly to meter poly
        left_fit_cr = np.array([left_fit[0] * xm_to_ym / self.ym_per_pix,
                                left_fit[1] * xm_to_ym,
                                left_fit[2] * self.xm_per_pix])
        right_fit_cr = np.array([right_fit[0] * xm_to_ym / self.ym_per_pix,
                                 right_fit[1] * xm_to_ym,
                                 right_fit[2] * self.xm_per_pix])

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = self.img_shape[0] * self.ym_per_pix

        # implement the calculation of R_curve (radius of curvature)
        left_curverad = (1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** (3 / 2) / abs(
            2 * left_fit_cr[0])
        right_curverad = (1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** (3 / 2) / abs(
            2 * right_fit_cr[0])

        return left_curverad, right_curverad
