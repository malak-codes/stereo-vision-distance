# Stereo Vision Distance Measurement

A Python-based stereo vision system for measuring real-world distances in images using camera calibration, epipolar geometry, and 3D triangulation.

## Overview

This project implements a complete stereo vision pipeline that allows users to:
- **Calibrate a camera** using checkerboard patterns from multiple perspectives
- **Measure real-world distances** by clicking on points in stereo image pairs
- **Automatically match corresponding points** between left and right images using epipolar geometry and template matching
- **Triangulate 3D coordinates** and transform them to real-world measurements in centimeters

## Project Structure

```
stereo-vision-distance/
├── src/
│   ├── stereo_distance_measurer.py    # Main interactive distance measurement tool
│   └── calibration.py                 # Camera calibration script
├── data/
│   ├── sample_images/
│   │   ├── box1.jpg                   # Sample stereo pair (right view)
│   │   └── box2.jpg                   # Sample stereo pair (left view)
│   ├── calibration_data/
│   │   └── camera_calibration_data.npz # Pre-computed camera calibration matrix
│   └── calibration_images/
│       ├── perspective1.jpeg through perspective8.jpeg
│       └── (Checkerboard patterns for camera calibration)
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Dependencies

Install the required packages:

```bash
pip install opencv-python numpy matplotlib
```

## Usage

### 1. Camera Calibration (Optional)

If you need to recalibrate the camera using your own checkerboard images:

```bash
cd src
python calibration.py
```

**What it does:**
- Reads all checkerboard images from `../data/calibration_images/`
- Detects corner points in each image using adaptive thresholding
- Computes the camera matrix (K) and distortion coefficients
- Saves the calibration data to `../data/calibration_data/camera_calibration_data.npz`

**Configuration:**
Edit the following variables in `calibration.py` if needed:
- `CHECKERBOARD_SIZE`: Number of inner corners (default: 10×7)
- `SQUARE_SIZE_CM`: Physical size of one checkerboard square (default: 2.5 cm)

### 2. Measure Distances

Run the interactive distance measurement tool:

```bash
cd src
python stereo_distance_measurer.py
```

**How to use:**

1. The script loads `box2.jpg` (left) and `box1.jpg` (right) from `../data/sample_images/`
2. It automatically computes the Essential Matrix and camera pose using SIFT features
3. A matplotlib window opens asking you to **click 4 reference points on the LEFT image**:
   - **Point 1**: Bottom-left corner of the object
   - **Point 2**: Top-left corner of the object
   - **Point 3**: Top-right corner of the object
   - **Point 4**: Back-top corner of the object (for depth)

4. For each left click, the script automatically:
   - Computes the epipolar line in the right image
   - Searches for the best match using normalized cross-correlation (NCC)
   - Triangulates the 3D coordinates
   - Displays the matched point with colored epilines

5. The script then:
   - Triangulates all 4 reference points
   - Computes a least-squares transformation matrix
   - Transforms the 3D points to real-world coordinates
   - Calculates distances (Height, Width, Depth) in centimeters
   - Prints the results

## Technical Details

### Algorithm Pipeline

1. **Feature Detection & Matching**
   - Uses SIFT (Scale-Invariant Feature Transform) to detect distinctive keypoints
   - Matches features between left and right images using FLANN (Fast Library for Approximate Nearest Neighbors)
   - Applies Lowe's ratio test (0.6 threshold) to filter ambiguous matches

2. **Epipolar Geometry**
   - Computes the Essential Matrix using RANSAC for robust estimation
   - Recovers camera pose (rotation R and translation t)
   - Computes the Fundamental Matrix from the Essential Matrix
   - Uses epipolar constraints to automatically find corresponding points

3. **Point Matching**
   - For each clicked point on the left image, computes the epipolar line on the right
   - Searches along the epipolar line using Normalized Cross-Correlation (NCC)
   - Selects the best match with the highest NCC score

4. **3D Triangulation**
   - Uses Direct Linear Transform (DLT) via `cv2.triangulatePoints`
   - Converts homogeneous coordinates to 3D Euclidean points

5. **Real-World Scaling**
   - Uses least-squares fitting to compute a transformation matrix
   - Maps triangulated 3D points to real-world coordinates in centimeters
   - Based on 4 reference points with known real-world dimensions

### Camera Calibration Parameters

The camera matrix (K) and distortion coefficients are stored in `camera_calibration_data.npz`:

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

dist = [k1, k2, p1, p2, k3]
```

Where:
- `fx, fy`: Focal lengths in pixels
- `cx, cy`: Principal point (image center)
- `k1, k2, k3`: Radial distortion coefficients
- `p1, p2`: Tangential distortion coefficients

## Sample Data

The repository includes:
- **Sample Images**: `box1.jpg` and `box2.jpg` - A stereo pair of a wooden nightstand
- **Calibration Images**: 8 checkerboard patterns captured from different perspectives
- **Pre-computed Calibration**: `camera_calibration_data.npz` - Ready-to-use camera parameters

## Customization

### Using Your Own Images

To measure distances in your own stereo image pairs:

1. Replace `box1.jpg` and `box2.jpg` in `data/sample_images/` with your stereo pair
2. Update the real-world coordinates in `stereo_distance_measurer.py` (lines 289-294):
   ```python
   real_world_coords = np.array([
       [0, 0, 0],           # Point 1 (origin)
       [0, height, 0],      # Point 2 (height)
       [width, height, 0],  # Point 3 (width)
       [width, height, depth]  # Point 4 (depth)
   ], dtype=np.float64)
   ```
3. Run `stereo_distance_measurer.py` as usual

### Recalibrating with Your Camera

If you want to recalibrate for a different camera:

1. Capture 8-10 images of a checkerboard pattern from different angles and distances
2. Place them in `data/calibration_images/` with `.jpeg` extension
3. Run `calibration.py`
4. The new calibration data will overwrite `camera_calibration_data.npz`

## Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **"No images found" error in calibration** | Ensure calibration images are in `data/calibration_images/` with `.jpeg` extension |
| **"box1.jpg / box2.jpg not found"** | Verify sample images are in `data/sample_images/` |
| **"Not enough matches for pose"** | Ensure stereo images have sufficient texture and overlap; try different images |
| **"No match found along the epipolar line"** | Select a more textured point with distinctive features |
| **Poor distance measurements** | Recalibrate the camera with better checkerboard images (ensure good lighting and coverage) |
| **Slow performance** | Reduce image resolution or adjust SIFT parameters |

## Requirements

- **OpenCV**: For image processing, SIFT feature detection, and camera calibration
- **NumPy**: For numerical computations and matrix operations
- **Matplotlib**: For interactive point clicking and visualization
- **Python 3.7+**: For language features and compatibility

## License

This project is licensed under the **MIT License** – see the LICENSE file for details.

## Author

Created as part of a computer vision and image processing course project.

## References

- OpenCV Documentation: https://docs.opencv.org/
- Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
- SIFT: Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
- Epipolar Geometry: https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html

## Contributing

Feel free to fork this repository and submit pull requests for improvements, bug fixes, or additional features.
