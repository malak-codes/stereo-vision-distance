import cv2
import numpy as np
import glob

# --- Configuration ---
CHECKERBOARD_SIZE = (10, 7)      # number of inner corners per row and column
SQUARE_SIZE_CM    = 2.5          # physical size of one square in cm
IMAGE_PATTERN     = '../data/calibration_images/perspective*.jpeg'  # glob pattern to locate images
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def main():
    objpoints = []  # 3D points in the world coordinate
    imgpoints = []  # 2D points in image plane
    
    # Prepare object points: (0,0,0), (1*square_size, 0, 0), ...
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_CM
    
    image_files = glob.glob(IMAGE_PATTERN)
    if not image_files:
        print(f"No images found matching {IMAGE_PATTERN}")
        return

    image_size = None
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: could not read image {fname}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            # Refine corner positions
            corners_subpix = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), CRITERIA
            )
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            print(f"{fname}: corners detected")
        else:
            print(f"{fname}: corners not found")

    if not objpoints:
        print("No valid views found – calibration aborted.")
        return
    
    # Calibrate camera
    error, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    print("\nCalibration completed.")
    print("Camera matrix (K):\n", K)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Average reprojection error:", error)
    
    # Save calibration data to file
    np.savez('../data/calibration_data/camera_calibration_data.npz', K=K, dist=dist_coeffs)
    print("Calibration data saved to '../data/calibration_data/camera_calibration_data.npz'."

if __name__ == "__main__":
    main()
