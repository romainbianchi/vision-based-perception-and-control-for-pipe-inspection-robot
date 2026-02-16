# Some code modified from https://github.com/Yacynte/Monocular-Visual-Odometry/blob/main(mvio)/visual_odometry.py

import numpy as np
import cv2

from sklearn import linear_model

class VisualOdometry:
    def __init__(self, calib_file):
        self.prev_image = None
        self.prev_pts = None
        self.lk_params = dict(winSize=(55, 55),
                              maxLevel=5,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.K, self.P = self.load_calib(calib_file)

    def load_calib(self, calib_file):
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        P = np.zeros((3, 4))
        for i in range(3):
            P[i] = list(map(float, lines[i].strip().split()))

        K = P[:, :3]
        
        return K, P

    
    def process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.prev_image is None:
            self.prev_image = image
            return np.eye(3), np.array([0, 0, 0]), image, image, False
        
        prev_gray = cv2.cvtColor(self.prev_image, cv2.COLOR_RGB2GRAY)

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=15,
                                            iterations=3,
                                            poly_n=5,
                                            poly_sigma=1.2,
                                            flags=0)

        # Generate a grid of points
        h, w = gray.shape
        step = 5  # Sampling step size (pixels)
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
        points_prev = np.stack((x, y), axis=-1).astype(np.float32)

        # Get flow vectors at those points
        flow_vectors = flow[y, x]  # Shape (N, 2)
        points_next = points_prev + flow_vectors

        # Filter based on flow magnitude
        magnitudes = np.linalg.norm(flow_vectors, axis=1)
        valid = magnitudes > 2.0  # Threshold in pixels
        good_prev = points_prev[valid]
        good_next = points_next[valid]

        # Reject outliers
        inliers, image_inliers_outliers = self.reject_outliers(image, good_prev, good_next)
        if len(inliers) > 2:
            good_prev = np.vstack([arr for arr, _ in inliers])
            good_next = np.vstack([arr for _, arr in inliers])

        if len(good_prev) < 100:
            self.prev_image = image
            return np.eye(3), np.array([0, 0, 0]), image, image, False

        # Estimate Essential matrix
        E, mask = cv2.findEssentialMat(good_prev, good_next, self.K, method=0, threshold=0.1)
        if E is None or mask is None:
            self.prev_image = image
            return np.eye(3), np.array([0, 0, 0]), image, image, False

        mask = mask.ravel().astype(bool)
        good_prev = good_prev[mask]
        good_next = good_next[mask]

        if len(good_prev) < 10 or len(good_next) < 10:
            self.prev_image = image
            return np.eye(3), np.array([0, 0, 0]), image, image, False

        R, t = self.decompose_essential_mat(E, good_prev, good_next)

        # Reject large rotations
        if self.large_rotation(R):
            R = np.eye(3)

        # Visualization
        image_with_arrows = image.copy()
        for (p, q) in zip(good_prev, good_next):
            x0, y0 = p.ravel()
            x1, y1 = q.ravel()
            cv2.arrowedLine(image_with_arrows, (int(x0), int(y0)), (int(x1), int(y1)),
                            color=(0, 255, 0), thickness=1, tipLength=0.3)

        # Update state
        self.prev_image = image
        self.prev_gray = gray

        return R, t, image_with_arrows, image_inliers_outliers, True
    
    def large_rotation(self, R):
        yaw = np.arctan2(R[0,2], R[2, 2])
        yaw_degrees = np.rad2deg(yaw)
        return abs(yaw_degrees) > 5.0

    def reject_outliers(self, image, pts1, pts2):

        img_height, img_width = image.shape[:2]

        # Divide image in 9 regions
        regions = []
        region_height = img_height // 5
        region_widhth =  img_width// 5
        for i in range(5):
            for j in range(5):
                x1 = j * region_widhth
                y1 = i * region_height
                x2 = (j + 1) * region_widhth
                y2 = (i + 1) * region_height
                regions.append((x1, y1, x2, y2))

        # for each region, keep only the points in that region and perform RANSAC to find the inliers
        inliers = []
        outliers = []
        ransac = linear_model.RANSACRegressor()
        for region in regions:
            x1, y1, x2, y2 = region
            mask = (pts1[:, 0] >= x1) & (pts1[:, 0] <= x2) & (pts1[:, 1] >= y1) & (pts1[:, 1] <= y2)
            pts1_region = pts1[mask]
            pts2_region = pts2[mask]

            if len(pts1_region) < 10 or len(pts2_region) < 10:
                continue

            ransac.fit(pts1_region, pts2_region)
            inlier_mask = ransac.inlier_mask_

            flow_vectors = pts2_region - pts1_region

            inliers_pts1 = pts1_region[inlier_mask]
            inliers_pts2 = pts2_region[inlier_mask]
            outliers_pts1 = pts1_region[~inlier_mask]
            outliers_pts2 = pts2_region[~inlier_mask]

            inliers.append((inliers_pts1, inliers_pts2))
            outliers.append((outliers_pts1, outliers_pts2))

        image_with_inliers_outliers = image.copy()

        # Draw inliers and outliers
        for inlier_pts1, inlier_pts2 in inliers:
            for (p, q) in zip(inlier_pts1, inlier_pts2):
                x0, y0 = p.ravel()
                x1, y1 = q.ravel()
                cv2.arrowedLine(image_with_inliers_outliers, (int(x0), int(y0)), (int(x1), int(y1)),
                                color=(0, 255, 0), thickness=1, tipLength=0.3)
        for outlier_pts1, outlier_pts2 in outliers:
            for (p, q) in zip(outlier_pts1, outlier_pts2):
                x0, y0 = p.ravel()
                x1, y1 = q.ravel()
                cv2.arrowedLine(image_with_inliers_outliers, (int(x0), int(y0)), (int(x1), int(y1)),
                                color=(0, 0, 255), thickness=1, tipLength=0.3)
                
        # Return the image with inliers and outliers
        return inliers, image_with_inliers_outliers
            
    
    def decompose_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(np.float32(self.P), np.float32(P), np.float32(q1.T), np.float32(q2.T))
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)
            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Compute pairwise distances
            dists_Q1 = np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)
            dists_Q2 = np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)

            # Avoid divide-by-zero by masking
            valid_mask = (dists_Q2 > 1e-6) & np.isfinite(dists_Q2) & np.isfinite(dists_Q1)

            if np.sum(valid_mask) == 0:
                relative_scale = 1.0  # fallback
            else:
                relative_scale = np.mean(dists_Q1[valid_mask] / dists_Q2[valid_mask])

            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)
        # t = t*z_c

        # Make a list of the different possible pairs
        pairs = [[R1, -t], [R1, t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, relative_scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(relative_scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        #print(right_pair_idx)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair

        return R1, t*relative_scale