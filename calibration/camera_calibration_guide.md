# Camera Calibration Guide

This guide explains how to calibrate your stereo camera system to generate the `camera_calibration_data.npz` file required by the stereo distance measurement system.

## Overview

Camera calibration is the process of determining the intrinsic parameters (focal length, principal point, distortion coefficients) of a camera. These parameters are essential for accurate 3D reconstruction and distance measurement.

## What You'll Need

- A printed checkerboard pattern (typically 8×6 or 9×7 squares)
- A calibration camera (the one you'll use for stereo imaging)
- OpenCV installed on your system
- Python 3.7+

## Step 1: Capture Calibration Images

1. Print a checkerboard pattern on a flat surface (A4 or larger paper recommended)
2. Mount the checkerboard on a rigid flat board
3. Take 20-30 images of the checkerboard from different angles:
   - Vary the distance (near and far)
   - Vary the angle (tilt, rotate)
   - Cover different regions of the image
4. Save images in a folder (e.g., `calibration_images/`)

## Step 2: Run the Calibration Script

Use the following Python script to calibrate your camera:

```python
import cv2
import numpy as np
import glob

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load calibration images
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    
    if ret:
        objpoints.append(objp)
        
        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# Calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Save calibration data
np.savez('camera_calibration_data.npz', K=K, dist=dist)
print(f"Calibration complete!")
print(f"Reprojection error: {ret:.3f}")
print(f"Camera matrix:\n{K}")
print(f"Distortion coefficients:\n{dist}")
```

## Step 3: Verify Calibration Quality

The reprojection error should be less than 1.0 pixel. If it's higher:
- Use more calibration images
- Ensure the checkerboard is flat and rigid
- Use higher resolution images
- Ensure good lighting conditions

## Checkerboard Dimensions

Common checkerboard sizes:
- **9×6**: 8×5 internal corners (recommended for most cameras)
- **8×6**: 7×5 internal corners
- **7×5**: 6×4 internal corners

Adjust the `objp` and `findChessboardCorners` parameters accordingly.

## Stereo Calibration (Optional)

If you want to calibrate a stereo pair specifically:

```python
# After individual calibration of both cameras
ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2,
    K1, dist1, K2, dist2,
    gray.shape[::-1],
    criteria=criteria,
    flags=cv2.STEREO_CALIB_FIX_INTRINSIC
)

# Save stereo calibration
np.savez('stereo_calibration.npz', 
         K1=K1, dist1=dist1, K2=K2, dist2=dist2,
         R=R, T=T, E=E, F=F)
```

## Troubleshooting

### Checkerboard Not Detected
- Ensure good lighting (no shadows or glare)
- Check that the checkerboard is fully visible in the image
- Try adjusting the checkerboard size in `findChessboardCorners`
- Ensure the image resolution is sufficient (at least 640×480)

### High Reprojection Error
- Use more calibration images (at least 30)
- Vary the angles and distances more
- Ensure the checkerboard is perfectly flat
- Check for lens distortion (use a wider range of images)

### Inconsistent Results
- Ensure all images are from the same camera with the same settings
- Avoid zooming (use fixed focal length)
- Ensure consistent lighting conditions

## Next Steps

Once you have the calibration file:
1. Place it in `data/calibration_data/`
2. Update the path in `stereo_distance_measurer.py`
3. Run the distance measurement script

## References

- OpenCV Camera Calibration: https://docs.opencv.org/master/d9/df8/tutorial_root.html
- Camera Calibration Theory: https://en.wikipedia.org/wiki/Camera_resectioning
