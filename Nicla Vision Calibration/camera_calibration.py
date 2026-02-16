# Camera calibration from OpenCV: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2 as cv
import os


def load_images(file_path):
    images_path = [os.path.join(file_path, file) for file in sorted(os.listdir(file_path)) if file.endswith('.jpg')]
    images = [cv.imread(image_path) for image_path in images_path]
    return images


def get_obj_img_points(criteria, calibration_pattern_size):
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((calibration_pattern_size[0]*calibration_pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:calibration_pattern_size[0],0:calibration_pattern_size[1]].T.reshape(-1,2)

    chessboard_squares_size_mm = 20
    objp = objp * chessboard_squares_size_mm
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    file_path = "calibration_images"
    images = load_images(file_path)
    
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, calibration_pattern_size, None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            cv.drawChessboardCorners(image, calibration_pattern_size, corners2, ret)
            cv.imshow('img', image)
            cv.waitKey(500)
    
    cv.destroyAllWindows()
    return objpoints, imgpoints


def main():

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_pattern_size = (5, 8)
    image_size = (320, 240)
    
    objpoints, imgpoints = get_obj_img_points(criteria, calibration_pattern_size)
    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("No chessboard corners found in the images.")
        return

    # Perform camera calibration
    rep_error, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    print(f' reprojection error: {rep_error}')

    # Camera parameters matrix
    K = mtx
    P = np.zeros((3, 4))
    P[:, :3] = K

    # # create calib.txt file and save P matrix in file
    # with open("calib.txt", "w") as f:
    #     for row in P:
    #         f.write("\t".join(map(str, row)) + "\n")
    # print("Camera calibration parameters saved to calib.txt") 

    # Create a calib.txt file and save P matrix on one line in the file
    with open("calib.txt", "w") as f:
        f.write("\t".join(map(str, P.flatten())) + "\n")
    print("Camera calibration parameters saved to calib.txt")    


if __name__ == '__main__':
    main()