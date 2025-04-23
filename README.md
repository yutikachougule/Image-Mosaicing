# Image Mosaicing Project

This project performs **image mosaicing** by stitching multiple overlapping images into a single panoramic image using feature detection, matching, and homography estimation.

## Dataset
- Images are loaded from the folder `DanaOffice\*`.
- `imageDatastore` is used to manage image loading.

## Method Overview

### 1. Preprocessing
- Convert images to grayscale.
- Apply Gaussian blur to suppress noise and improve corner detection.

### 2. Feature Detection
- Use Harris Corner Detection to extract features from each image.

### 3. Feature Description
- Extract image patches (windows) around each detected corner.

### 4. Feature Matching
- Perform feature matching using **Normalized Cross-Correlation (NCC)**.
- Threshold the matches based on NCC score.

### 5. Homography Estimation
- Estimate homography between matched features using **RANSAC** to eliminate outliers.
- Compute cumulative transformations relative to a reference (central) image.

### 6. Image Warping
- Warp each image onto the panorama canvas using the estimated homography.
- Use bilinear interpolation with `interp2`.

### 7. Image Blending
- Combine warped images by averaging overlapping pixels and taking the maximum otherwise.

## File Structure

- `getHarrisCorners` - Extracts Harris corners.
- `getFeatures` - Extracts fixed-size patch features around corners.
- `matchFeaturesNCC` - Matches features using NCC.
- `getInliersRANSAC` - Uses RANSAC to compute best inliers for homography.
- `estimateBestHomography` - Computes homography from inliers.

## Output
- Each warped image is displayed individually.
- A final panoramic image is constructed by blending all warped images.

## Requirements
- MATLAB (Image Processing Toolbox)

## To Run

Place all image files in `DanaOffice` directory, then run:

```matlab
%% Project 2: Image Mosaicing
% (Script will handle everything from feature detection to image stitching)
