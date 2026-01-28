import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# =============================================================
# Stereo Measurement (strict epipolar-constrained auto-matching)
# =============================================================
# You click ONLY on the LEFT image.
# For each left click, we automatically find the RIGHT correspondence by:
#   1) Computing the epipolar line l' in the RIGHT view from E (and K).
#   2) Collecting RIGHT keypoints whose perpendicular distance to l' <= eps px.
#   3) Among those, taking the SIFT descriptor best-match to the *closest* LEFT
#      keypoint to your click (ratio-tested earlier).
#   4) If none pass, we fall back to 1D template matching (NCC) along a thin
#      band around the epiline, then subpixel refine with cornerSubPix.
# We also draw HIGH-CONTRAST epilines and put a zoom-in crop for each match.
#
# References (official docs):
# - Epipolar geometry / Essential / recoverPose / triangulate / undistort:
#   https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
# - Tutorial (E, F, epilines):
#   https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
# - SIFT / FLANN / drawing / LK / cornerSubPix:
#   https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
#   https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
#   https://docs.opencv.org/4.x/d7/d8b/tutorial_py_lucas_kanade.html

# -------------------------------
# Utilities
# -------------------------------

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def undistort_norm_point(pt_xy: Tuple[float, float], K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    uv = np.array([[pt_xy]], dtype=np.float32)  # (1,1,2)
    norm = cv2.undistortPoints(uv, K, dist)
    return norm.reshape(2, 1).astype(np.float32)


def triangulate_pair(ptL_xy: Tuple[float, float], ptR_xy: Tuple[float, float],
                     P1: np.ndarray, P2: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pL = undistort_norm_point(ptL_xy, K, dist)
    pR = undistort_norm_point(ptR_xy, K, dist)
    X_h = cv2.triangulatePoints(P1, P2, pL, pR)
    X = (X_h[:3] / X_h[3]).reshape(3)
    return X.astype(np.float64)


def click_points(img, n: int, title: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(title)

    def on_click(event):
        if not event.inaxes:
            return
        pts.append((float(event.xdata), float(event.ydata)))
        ax.plot(event.xdata, event.ydata, 'ro')
        ax.figure.canvas.draw()
        if len(pts) == n:
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    if len(pts) != n:
        raise RuntimeError(f"You must click exactly {n} points.")
    return pts

# -------------------------------
# Epipolar helpers & visualization
# -------------------------------

def fundamental_from_essential(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    Kinv = np.linalg.inv(K)
    return Kinv.T @ E @ Kinv


def epiline_for_point_right(F: np.ndarray, ptL_xy: Tuple[float, float]) -> np.ndarray:
    # line l' = F * p (homogeneous); returns (a,b,c) s.t. a x + b y + c = 0
    p = np.array([ptL_xy[0], ptL_xy[1], 1.0], dtype=np.float64)
    l = F @ p
    return l / np.linalg.norm(l[:2])


def point_to_line_distance(px: float, py: float, line_abc: np.ndarray) -> float:
    a, b, c = line_abc
    return abs(a*px + b*py + c)


def draw_epilines(imgR: np.ndarray, lines_abc: list, thickness: int = 2):
    out = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    h, w = imgR.shape
    for i, (a,b,c) in enumerate(lines_abc):
        if abs(b) < 1e-9:
            continue
        y0 = int(round(-c / b))
        yw = int(round((-a * (w-1) - c) / b))
        cv2.line(out, (0,y0), (w-1,yw), colors[i % len(colors)], thickness)
    return out



# -------------------------------
# Right-point search (strict epipolar + descriptor + NCC fallback)
# -------------------------------

def auto_right_points_strict(imgL, imgR, left_pts: List[Tuple[float, float]], F: np.ndarray) -> List[Tuple[float, float]]:
    rights = []
    window_size = 100  # Size of the window (50x50)
    step_size = 5     # Step size for moving the window

    for ptL in left_pts:
        line = epiline_for_point_right(F, ptL)
        a, b, c = line

        # Extract the 50x50 window around the left point
        xL, yL = int(ptL[0]), int(ptL[1])
        hL, wL = imgL.shape
        left_window = imgL[max(0, yL - window_size // 2):min(hL, yL + window_size // 2),
                           max(0, xL - window_size // 2):min(wL, xL + window_size // 2)].astype(np.float32)

        if left_window.size == 0:
            raise RuntimeError("Left window is empty. Ensure the selected point is within bounds.")

        # Normalize the left window
        left_window = (left_window - left_window.mean()) / (left_window.std() + 1e-8)

        best_match = None
        best_score = -1.0

        # Scan along the epiline in the right image
        hR, wR = imgR.shape
        for xR in range(0, wR, step_size):
            yR = int(round((-a * xR - c) / b)) if abs(b) > 1e-6 else yL
            if yR - window_size // 2 < 0 or yR + window_size // 2 >= hR or xR - window_size // 2 < 0 or xR + window_size // 2 >= wR:
                continue

            # Extract the 50x50 window around the current point in the right image
            right_window = imgR[yR - window_size // 2:yR + window_size // 2,
                                xR - window_size // 2:xR + window_size // 2].astype(np.float32)

            if right_window.shape != left_window.shape:
                continue

            # Normalize the right window
            right_window = (right_window - right_window.mean()) / (right_window.std() + 1e-8)

            # Compute the normalized cross-correlation (NCC)
            score = float((left_window * right_window).sum())
            if score > best_score:
                best_score = score
                best_match = (float(xR), float(yR))

        if best_match is None:
            raise RuntimeError("No match found along the epipolar line. Try selecting a more textured point.")
        rights.append(best_match)

    return rights


def ncc_along_epiline(imgL, imgR, ptL: Tuple[float,float], line_abc: np.ndarray,
                      half_win: int = 8, search_half_band: int = 25) -> Tuple[float,float] | None:
    """Simple normalized cross-correlation in a thin band around the epiline."""
    x, y = int(round(ptL[0])), int(round(ptL[1]))
    h, w = imgL.shape
    x0, x1 = max(0, x-half_win), min(w, x+half_win+1)
    y0, y1 = max(0, y-half_win), min(h, y+half_win+1)
    tpl = imgL[y0:y1, x0:x1].astype(np.float32)
    if tpl.size == 0:
        return None
    # Build mask points near epiline in right image
    a,b,c = line_abc
    best = None
    best_val = -1.0
    # Scan a rectangle around the expected y ~ (-a x - c)/b
    for xr in range(max(half_win, 0), min(w-half_win-1, w)):
        yr = int(round((-a*xr - c)/b)) if abs(b) > 1e-6 else y
        for dy in range(-search_half_band, search_half_band+1):
            yy = yr + dy
            if yy - half_win < 0 or yy + half_win + 1 >= h:
                continue
            roi = imgR[yy-half_win:yy+half_win+1, xr-half_win:xr+half_win+1].astype(np.float32)
            if roi.shape != tpl.shape:
                continue
            # NCC
            t = tpl - tpl.mean(); r = roi - roi.mean()
            denom = (np.linalg.norm(t)*np.linalg.norm(r) + 1e-6)
            val = float((t*r).sum()/denom)
            if val > best_val:
                best_val = val
                best = (float(xr), float(yy))
    return best

def find_best_match_along_epipolar_line(ptL, kp2, des2, F, left_descriptor, img2):
    # Compute the epipolar line
    x1_h = np.array([ptL[0], ptL[1], 1.0])  # Homogeneous coordinates
    line_r = F @ x1_h
    a, b, c = line_r

    # Sample points along the epipolar line
    height, width = img2.shape
    sampled_points = []
    for x in range(width):
        y = int(-(a * x + c) / b)
        if 0 <= y < height:
            sampled_points.append((x, y))

    # Extract descriptors for sampled points
    sampled_descriptors = []
    for x, y in sampled_points:
        closest_kp = min(kp2, key=lambda kp: np.linalg.norm(np.array(kp.pt) - np.array([x, y])))
        sampled_descriptors.append(des2[closest_kp.queryIdx])

    # Compare descriptors using NCC
    def normalized_cross_correlation(desc1, desc2):
        desc1 = (desc1 - np.mean(desc1)) / (np.std(desc1) + 1e-8)
        desc2 = (desc2 - np.mean(desc2)) / (np.std(desc2) + 1e-8)
        return np.sum(desc1 * desc2) / len(desc1)

    best_match = None
    best_score = -1
    for sampled_desc in sampled_descriptors:
        score = normalized_cross_correlation(left_descriptor, sampled_desc)
        if score > best_score:
            best_score = score
            best_match = sampled_desc

    return best_match

# # -------------------------------
# # Main
# # -------------------------------
if __name__ == "__main__":
    img1 = cv2.imread('../data/sample_images/box2.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../data/sample_images/box1.jpg', cv2.IMREAD_GRAYSCALE)
    assert img1 is not None and img2 is not None, "box1.jpg / box2.jpg not found (or paths incorrect)."

    calib = np.load('../data/calibration_data/camera_calibration_data.npz')
    K = calib['K'].astype(np.float64)
    dist = calib['dist']

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        raise RuntimeError("Failed to compute SIFT descriptors. Ensure images have enough texture.")

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < 0.6 * n.distance]
    if len(good) < 12:
        raise RuntimeError(f"Not enough matches for pose (have {len(good)}; need >= 12).")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    E, inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("findEssentialMat failed. Check matches and calibration.")
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t])

    F = fundamental_from_essential(E, K)

    # ------------- Click 4 reference points on LEFT only -------------
    left4 = click_points(img1, 4, "LEFT image: Click 4 points (Bottom, Top, Other-Top, Back)")
    right4 = auto_right_points_strict(img1, img2, left4,F)

    # Visualization: high-contrast epilines + colored points + zooms
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Yellow
    epi_lines = [epiline_for_point_right(F, p) for p in left4]
    right_epishow = draw_epilines(img2, epi_lines, thickness=2)
    for i, (x, y) in enumerate(right4):
        color = colors[i % len(colors)]
        cv2.circle(right_epishow, (int(round(x)), int(round(y))), 5, color, -1)
        cv2.putText(right_epishow, f"ref{i+1}", (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    plt.figure(); plt.imshow(cv2.cvtColor(right_epishow, cv2.COLOR_BGR2RGB)); plt.title('RIGHT: epilines + auto matches (colored)'); plt.axis('off'); plt.show()

    # Define real-world coordinates for the clicked points
    real_world_coords = np.array([
        [0, 0, 0],       # First point: Origin
        [0, 43, 0],      # Second point: Along Y-axis
        [55, 43, 0],     # Third point: Along X-axis
        [55, 43, 45]     # Fourth point: Along Z-axis
    ], dtype=np.float64)

    # Triangulate 4 reference points
    X_bottom = triangulate_pair(left4[0], right4[0], P1, P2, K, dist)
    X_top    = triangulate_pair(left4[1], right4[1], P1, P2, K, dist)
    X_other  = triangulate_pair(left4[2], right4[2], P1, P2, K, dist)
    X_back   = triangulate_pair(left4[3], right4[3], P1, P2, K, dist)

    # Stack triangulated points
    triangulated_points = np.vstack([X_bottom, X_top, X_other, X_back])

    # Compute scaling and transformation matrix
    A = np.linalg.lstsq(triangulated_points, real_world_coords, rcond=None)[0]

    # Transform triangulated points to real-world coordinates
    transformed_points = triangulated_points @ A

    # Output transformed points
    print("Transformed Points:")
    for i, point in enumerate(transformed_points):
        print(f"Point {i+1}: {point}")

    # Compute distances between transformed points
    transformed_height = np.linalg.norm(transformed_points[1] - transformed_points[0])
    transformed_width = np.linalg.norm(transformed_points[2] - transformed_points[1])
    transformed_depth = np.linalg.norm(transformed_points[3] - transformed_points[2])

    print("Transformed Distances:")
    print(f"Height: {transformed_height:.2f} cm")
    print(f"Width: {transformed_width:.2f} cm")
    print(f"Depth: {transformed_depth:.2f} cm")

    # # ------------- Click 2 measurement points on LEFT only -------------
    # left2 = click_points(img1, 2, "LEFT image: Click the TWO measurement points (e.g., on the floor)")
    # right2 = auto_right_points_strict(img1, img2, left2, F)

    # epi_lines2 = [epiline_for_point_right(F, p) for p in left2]
    # right_epishow2 = draw_epilines(img2, epi_lines2, thickness=2)
    # for i, (x,y) in enumerate(right2):
    #     cv2.circle(right_epishow2, (int(round(x)), int(round(y))), 5, (0,0,255), -1)
    #     cv2.putText(right_epishow2, f"m{i+1}", (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
    # plt.figure(); plt.imshow(cv2.cvtColor(right_epishow2, cv2.COLOR_BGR2RGB)); plt.title('RIGHT: epilines + auto matches (measurement)'); plt.axis('off'); plt.show()

    # X1 = triangulate_pair(left2[0], right2[0], P1, P2, K, dist)
    # X1_scaled = np.array([X1[0]*s_w, X1[1]*s_h, X1[2]*s_d])
    # X2 = triangulate_pair(left2[1], right2[1], P1, P2, K, dist)
    # X2_scaled = np.array([X2[0]*s_w, X2[1]*s_h, X2[2]*s_d])
    # distance_cm = l2(X1_scaled, X2_scaled)

    # print("Measured Distance ---")
    # print(f"3D distance between clicked points (scaled to cm): {distance_cm:.2f} cm")

    # # Define scaling factors based on real-world dimensions and measured distances
    # scale_y = real_height_cm / np.linalg.norm(X_top - X_bottom)
    # scale_x = real_width_cm / np.linalg.norm(X_other - X_top)
    # scale_z = real_depth_cm / np.linalg.norm(X_back - X_other)

    # # Apply scaling directly to the image instead of using the median
    # scaled_X_bottom = X_bottom * scale_y
    # scaled_X_top = X_top * scale_y
    # scaled_X_other = X_other * scale_x
    # scaled_X_back = X_back * scale_z

    # # Output scaled points
    # print("Scaled Points:")
    # print(f"Bottom: {scaled_X_bottom}")
    # print(f"Top: {scaled_X_top}")
    # print(f"Other: {scaled_X_other}")
    # print(f"Back: {scaled_X_back}")

    # # Compute distances between scaled points
    # scaled_height = np.linalg.norm(scaled_X_top - scaled_X_bottom)
    # scaled_width = np.linalg.norm(scaled_X_other - scaled_X_top)
    # scaled_depth = np.linalg.norm(scaled_X_back - scaled_X_other)

    # print("Scaled Distances:")
    # print(f"Height: {scaled_height:.2f} cm")
    # print(f"Width: {scaled_width:.2f} cm")
    # print(f"Depth: {scaled_depth:.2f} cm")

    # # ---------------- Tile-Based Scaling ----------------
    # # Step 1: Click the 4 corners of a tile
    # tile_corners = click_points(img1, 4, "Click the 4 corners of a tile on the floor")

    # # Step 2: Define real-world tile coordinates
    # tile_size = 33.0  # Tile size in cm
    # real_world_tile = np.array([
    #     [0, 0],               # Bottom-left corner
    #     [tile_size, 0],       # Bottom-right corner
    #     [tile_size, tile_size],  # Top-right corner
    #     [0, tile_size]        # Top-left corner
    # ], dtype=np.float32)

    # # Step 3: Compute the homography
    # tile_corners_np = np.array(tile_corners, dtype=np.float32)
    # H, _ = cv2.findHomography(tile_corners_np, real_world_tile)

    # # Step 4: Click two points to measure distance
    # clicked_points = click_points(img1, 2, "Click two points to measure distance")
    # clicked_points_np = np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)

    # # Step 5: Transform points to real-world coordinates
    # real_world_points = cv2.perspectiveTransform(clicked_points_np, H)

    # # Step 6: Compute the scaled distance
    # point1, point2 = real_world_points[0][0], real_world_points[1][0]
    # distance = np.linalg.norm(point1 - point2)
    # print(f"Measured distance: {distance:.2f} cm")