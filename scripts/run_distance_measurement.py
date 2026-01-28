#!/usr/bin/env python3
"""
Batch Distance Measurement Utility

This script allows you to measure distances between multiple point pairs
in stereo images without manual interaction.

Usage:
    python run_distance_measurement.py --left <left_image> --right <right_image> \
                                       --calib <calibration.npz> \
                                       --points <points.txt>

Points file format (one pair per line):
    x1_left,y1_left,x1_right,y1_right
    x2_left,y2_left,x2_right,y2_right
    ...
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def undistort_norm_point(pt_xy, K, dist):
    """Undistort and normalize a point."""
    uv = np.array([[pt_xy]], dtype=np.float32)
    norm = cv2.undistortPoints(uv, K, dist)
    return norm.reshape(2, 1).astype(np.float32)


def triangulate_pair(ptL_xy, ptR_xy, P1, P2, K, dist):
    """Triangulate a point pair from two views."""
    pL = undistort_norm_point(ptL_xy, K, dist)
    pR = undistort_norm_point(ptR_xy, K, dist)
    X_h = cv2.triangulatePoints(P1, P2, pL, pR)
    X = (X_h[:3] / X_h[3]).reshape(3)
    return X.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description='Batch distance measurement from stereo images')
    parser.add_argument('--left', required=True, help='Path to left image')
    parser.add_argument('--right', required=True, help='Path to right image')
    parser.add_argument('--calib', required=True, help='Path to calibration .npz file')
    parser.add_argument('--points', required=True, help='Path to points file')
    parser.add_argument('--output', default='distances.txt', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load images
    img1 = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        return 1
    
    # Load calibration
    calib = np.load(args.calib)
    K = calib['K'].astype(np.float64)
    dist = calib['dist']
    
    # Compute essential matrix and pose
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < 0.6 * n.distance]
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t])
    
    # Read point pairs
    results = []
    try:
        with open(args.points, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    coords = list(map(float, line.split(',')))
                    if len(coords) != 4:
                        print(f"Warning: Line {i} has {len(coords)} values, expected 4")
                        continue
                    
                    x1_left, y1_left, x1_right, y1_right = coords
                    
                    # Triangulate
                    X = triangulate_pair((x1_left, y1_left), (x1_right, y1_right), P1, P2, K, dist)
                    
                    # Compute distance from origin
                    distance = np.linalg.norm(X)
                    
                    results.append({
                        'line': i,
                        'point': (x1_left, y1_left),
                        'distance': distance,
                        'coords_3d': X
                    })
                    
                    print(f"Point {i}: Distance = {distance:.2f} cm, 3D coords = {X}")
                
                except ValueError as e:
                    print(f"Error parsing line {i}: {e}")
    
    except FileNotFoundError:
        print(f"Error: Points file not found: {args.points}")
        return 1
    
    # Save results
    with open(args.output, 'w') as f:
        f.write("Point,Distance(cm),X,Y,Z\n")
        for r in results:
            x, y, z = r['coords_3d']
            f.write(f"{r['line']},{r['distance']:.2f},{x:.2f},{y:.2f},{z:.2f}\n")
    
    print(f"\nResults saved to {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
