# Stereo Vision Distance Measurement

A comprehensive Python implementation for measuring real-world distances from stereo image pairs using epipolar geometry, SIFT feature matching, and 3D triangulation.

## Overview

This project implements a sophisticated stereo vision pipeline that enables accurate measurement of real-world distances from two calibrated camera views. By combining SIFT feature detection, RANSAC-based pose estimation, and least-squares coordinate transformation, the system can measure distances in centimeters with high precision.

## Key Features

- **Automatic Feature Matching**: Uses SIFT descriptors with FLANN-based matching and Lowe's ratio test for robust correspondence detection.
- **Epipolar Geometry**: Implements the essential matrix computation and recovery of camera pose (rotation and translation).
- **3D Triangulation**: Computes 3D world coordinates from matched image points using the Direct Linear Transform (DLT) method.
- **Real-World Scaling**: Applies a least-squares transformation matrix to convert triangulated points into real-world centimeters.
- **Interactive Point Selection**: Allows users to click on image points and automatically finds corresponding matches in the second image using epipolar constraints.
- **Robust Matching**: Combines SIFT descriptors with template matching (NCC) and subpixel refinement for reliable correspondence.

## Project Structure

```
stereo-vision-distance/
├── src/
│   └── stereo_distance_measurer.py    # Main implementation
├── calibration/
│   └── camera_calibration_guide.md    # Instructions for camera calibration
├── data/
│   ├── sample_images/
│   │   ├── box1.jpg                   # Sample stereo pair (view 1)
│   │   └── box2.jpg                   # Sample stereo pair (view 2)
│   └── calibration_data/
│       └── camera_calibration_data.npz # Pre-computed calibration matrix
├── scripts/
│   └── run_distance_measurement.py    # Utility script for batch processing
├── docs/
│   ├── theory.md                      # Mathematical background
│   └── troubleshooting.md             # Common issues and solutions
├── README.md                          # This file
└── LICENSE                            # MIT License
```

## Installation

### Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib (optional, for visualization)

### Setup

```bash
# Clone the repository
git clone https://github.com/malak-codes/stereo-vision-distance.git
cd stereo-vision-distance

# Install dependencies
pip install opencv-python numpy matplotlib

# Verify installation
python -c "import cv2; print(cv2.__version__)"
```

## Usage

### Basic Usage

The main script is interactive and allows you to click on points in the left image. The system automatically finds the corresponding point in the right image and calculates the 3D distance.

```bash
cd src
python stereo_distance_measurer.py
```

### Step-by-Step Guide

1. **Prepare Your Images**: Ensure you have a stereo pair of images (left and right views) taken with the same calibrated camera.

2. **Prepare Calibration Data**: You need a `.npz` file containing:
   - `K`: Camera intrinsic matrix (3×3)
   - `dist`: Distortion coefficients

3. **Update Image Paths**: Modify the image paths in `stereo_distance_measurer.py` (lines 239-240):
   ```python
   img1 = cv2.imread('path/to/left_image.jpg', cv2.IMREAD_GRAYSCALE)
   img2 = cv2.imread('path/to/right_image.jpg', cv2.IMREAD_GRAYSCALE)
   ```

4. **Load Calibration Data**: Update the calibration file path (line 243):
   ```python
   calib = np.load('path/to/camera_calibration_data.npz')
   ```

5. **Run the Script**: Execute the main script and follow the on-screen instructions to click points.

### Example with Sample Data

```bash
# Navigate to the source directory
cd src

# Run with sample images (already configured)
python stereo_distance_measurer.py
```

The sample images show a wooden nightstand, and the pre-computed calibration data is set up for these specific images.

## How It Works

### 1. Feature Detection and Matching

The system uses SIFT (Scale-Invariant Feature Transform) to detect keypoints in both images. These keypoints are matched using a FLANN-based matcher with Lowe's ratio test to filter out ambiguous matches.

```python
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
```

### 2. Pose Estimation

The essential matrix is computed from matched points using RANSAC, which is then decomposed to recover the camera rotation (R) and translation (t) between the two views.

```python
E, inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
```

### 3. Epipolar Geometry

For each point clicked by the user in the left image, the system computes the epipolar line in the right image. This line constrains where the corresponding point must lie.

```python
def epiline_for_point_right(F, ptL_xy):
    p = np.array([ptL_xy[0], ptL_xy[1], 1.0], dtype=np.float64)
    l = F @ p
    return l / np.linalg.norm(l[:2])
```

### 4. Automatic Correspondence Finding

The system searches along the epipolar line in the right image to find the best matching point using normalized cross-correlation (NCC) on local image patches.

### 5. 3D Triangulation

Once a correspondence is found, the 3D world coordinates are computed using the Direct Linear Transform:

```python
X_h = cv2.triangulatePoints(P1, P2, pL, pR)
X = (X_h[:3] / X_h[3]).reshape(3)
```

### 6. Real-World Scaling

A transformation matrix is computed from reference points to map the triangulated 3D points into real-world centimeters:

```python
A = np.linalg.lstsq(np.vstack(X_pts), real_world_coords, rcond=None)[0]
```

## Camera Calibration

To use this system with your own camera, you need to calibrate it first. See `calibration/camera_calibration_guide.md` for detailed instructions on how to:

1. Capture calibration images (checkerboard pattern)
2. Run OpenCV's calibration algorithm
3. Generate the `.npz` calibration file

## Troubleshooting

### "Not enough matches for pose estimation"
- Ensure both images have sufficient texture and overlap
- Try adjusting the SIFT parameters or the ratio test threshold (currently 0.6)

### "Failed to compute SIFT descriptors"
- Check that the images are valid and contain enough detail
- Ensure images are not too small or too blurry

### Inaccurate distance measurements
- Verify that the calibration data is correct for your camera
- Ensure the stereo pair was taken with the same camera and settings
- Check that the reference points used for scaling are accurate

## Mathematical Background

For a detailed explanation of the algorithms used, including the mathematics of epipolar geometry, essential matrices, and triangulation, see `docs/theory.md`.

## Performance Notes

- **Processing Time**: ~1-2 seconds per point pair on modern hardware
- **Accuracy**: Typically within 1-5% of actual distance (depends on calibration quality and baseline)
- **Baseline**: Larger baseline between cameras improves depth accuracy

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

- Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press.
- Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision*, 60(2), 91-110.
- OpenCV Documentation: https://docs.opencv.org/

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue on the GitHub repository.
