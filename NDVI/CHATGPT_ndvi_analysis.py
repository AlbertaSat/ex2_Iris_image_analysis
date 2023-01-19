import cv2
import numpy as np

def align_images(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors using SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match keypoints using FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Select good matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Compute the homography matrix
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Align the images using the homography matrix
    aligned_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    return aligned_img1

def calculate_ndvi(red, nir):
    # Convert images to float32
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)

    # Calculate NDVI
    ndvi = (nir - red) / (nir + red)

    return ndvi

# Example usage
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")
aligned_img1 = align_images(img1, img2)
red = aligned_img1[:, :, 2]
nir = img2[:, :, 3]
ndvi = calculate_ndvi(red, nir)
print(ndvi)
