import numpy as np
import cv2


class Pipeline:
    def __init__(self):
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

    def threshold(self, img, g_thresh=(20, 100), s_thresh=(170, 255)):
        """Use sobel gradient threshold and saturation channel threshold to find edges."""
        img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= g_thresh[0]) & (scaled_sobel <= g_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # Get mask
        mask = np.zeros_like(s_channel)
        mask[(sxbinary == 1) | (s_binary == 1)] = 1
        return color_binary, mask

    def warp_img(self, img):
        warped = cv2.warpPerspective(src=img, M=self.perspective_M,
                                     dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return warped

    def warp_img_back(self, warped):
        img = cv2.warpPerspective(src=warped, M=self.perspective_Minv,
                                  dsize=(warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)
        return img
