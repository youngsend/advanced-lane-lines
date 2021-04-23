import numpy as np
import cv2
import glob
import os

CORNER_SIZE = (9, 6)


class CameraCalibration:
    def __init__(self, rel_cal_img_folder_path):
        self.project_path = os.path.abspath(os.curdir)
        self.cal_img_names = glob.glob(self.project_path + '/' + rel_cal_img_folder_path + '*.jpg')
        self.obj_points, self.img_points, self.image_size = self.collect_object_image_points()
        # this is camera calibration, therefore imageSize is fixed.
        self.ret, self.camera_mtx, self.dist_coeffs, self.rvecs, self.tvecs \
            = cv2.calibrateCamera(objectPoints=self.obj_points,
                                  imagePoints=self.img_points,
                                  imageSize=self.image_size,
                                  cameraMatrix=None,
                                  distCoeffs=None)

    def collect_object_image_points(self):
        """Collect 3D points and 2D image points from chessboard images."""
        # arrays to store object points and image points from all the images
        obj_points = []
        img_points = []

        # prepare object points, like (0,0,0),(1,0,0),(2,0,0)....,(8,5,0)
        obj_p = np.zeros((CORNER_SIZE[0] * CORNER_SIZE[1], 3), np.float32)
        # np.mgrid[0:9, 0:6] shape: (2,9,6), its T's shape: (6,9,2), after reshape: (54,2)
        obj_p[:, :2] = np.mgrid[0:CORNER_SIZE[0], 0:CORNER_SIZE[1]].T.reshape(-1, 2)  # x, y coordinates

        image_size = None
        for image_name in self.cal_img_names:
            img = cv2.imread(image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # record the imageSize, which will be used in calibrateCamera function.
            if not image_size:
                image_size = gray.shape[::-1]

            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, CORNER_SIZE, None)

            # if corners are found, add object points, image points
            if ret:
                img_points.append(corners)
                obj_points.append(obj_p)
        return obj_points, img_points, image_size

    def undistort(self, img):
        """Undistort image using calibration parameters."""
        # in order to use this function, CameraCalibration object has to be created, which means camera calibration has
        # to be done again. Maybe saving cameraMatrix and distCoeffs is better.
        dst = cv2.undistort(src=img,
                            cameraMatrix=self.camera_mtx,
                            distCoeffs=self.dist_coeffs,
                            dst=None,
                            newCameraMatrix=self.camera_mtx)
        return dst
