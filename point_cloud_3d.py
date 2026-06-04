"""
3D Point Cloud Reconstruction from Video
=========================================
Step 1: Dense point cloud generation from MiDaS depth maps.
Step 2: 3D bounding boxes around YOLO-detected vehicles.
Step 3: Ground plane grid, binary PLY export, progress tracking.
Back-projects pixels into 3D (X, Y, Z), detects vehicles,
and exports colored point clouds with wireframe cuboids as .ply files.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ─── Configuration ───────────────────────────────────────────────────
VIDEO_PATH = "ADAS.mp4"
OUTPUT_DIR = "output/pointclouds"
FRAME_INTERVAL = 30          # Save a .ply every N frames (e.g., every 30 frames ≈ 1 per second)
DOWNSAMPLE = 4               # Sample every Nth pixel (4 = 1/4 resolution, keeps file size manageable)
MAX_DEPTH_M = 30.0           # Clip points beyond this depth (meters)
MIN_DEPTH_M = 1.0            # Clip points closer than this (meters)

# Camera intrinsics (matching detect_and_hud.py)
FRAME_W = 1280
FRAME_H = 720
F_X = 1700.0                 # Focal length X (pixels)
F_Y = 1700.0                 # Focal length Y (pixels) 
C_X = 640.0                  # Principal point X
C_Y = 360.0                  # Principal point Y
H_CAM = 1.5                  # Camera height above ground (meters)


def write_ply(filepath, points, colors):
    """
    Write a point cloud to a binary .ply file (much faster and smaller than ASCII).
    
    Args:
        filepath: Output .ply file path
        points:   Nx3 numpy array of (X, Y, Z) coordinates
        colors:   Nx3 numpy array of (R, G, B) values [0-255]
    """
    import struct
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        # Pack all data as binary
        for i in range(n):
            x, y, z = points[i]
            r, g, b = int(colors[i][0]), int(colors[i][1]), int(colors[i][2])
            f.write(struct.pack('<fffBBB', float(x), float(y), float(z), r, g, b))
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved {n:,} points ({file_size_mb:.1f} MB) → {filepath}")


def generate_ground_grid(x_range=(-15, 15), z_range=(-2, 32), spacing=0.5):
    """
    Generate a dense ground plane grid (visible surface like reference image).
    Lines are drawn at regular intervals along X and Z axes at Y = H_CAM.
    
    Args:
        x_range: (min_x, max_x) in meters
        z_range: (min_z, max_z) in meters
        spacing: distance between grid lines in meters (0.5 = dense)
    
    Returns:
        grid_points: Nx3 array
        grid_colors: Nx3 array (dark gray)
    """
    points = []
    colors = []
    grid_color = (50, 50, 50)  # Dark gray
    y_ground = H_CAM  # Ground level
    pts_per_line = 150  # Dense for visible surface
    
    # Lines along Z axis (at each X position)
    for x in np.arange(x_range[0], x_range[1] + spacing, spacing):
        for z in np.linspace(z_range[0], z_range[1], pts_per_line):
            points.append([x, y_ground, z])
            colors.append(grid_color)
    
    # Lines along X axis (at each Z position)
    for z in np.arange(z_range[0], z_range[1] + spacing, spacing):
        for x in np.linspace(x_range[0], x_range[1], pts_per_line):
            points.append([x, y_ground, z])
            colors.append(grid_color)
    
    return np.array(points), np.array(colors)


def depth_to_point_cloud(depth_metric, frame_rgb, downsample=3):
    """
    Project road pixels into a FLAT ground-plane point cloud.
    
    Instead of creating a perspective frustum (vertical wall), all road points
    are flattened to Y = H_CAM (ground level). Only X and Z vary, creating
    a flat road surface viewed from above — matching the reference image style.
    
    Args:
        depth_metric: HxW array of metric depth values (meters)
        frame_rgb:    HxW x3 array of RGB pixel colors
        downsample:   Sample every Nth pixel
    
    Returns:
        points: Nx3 array of (X, Y, Z)
        colors: Nx3 array of (R, G, B)
    """
    h, w = depth_metric.shape
    
    # Only sample road region (below horizon)
    HORIZON_Y = 340
    
    vs = np.arange(HORIZON_Y, h, downsample)
    us = np.arange(0, w, downsample)
    uu, vv = np.meshgrid(us, vs)
    
    depth_sampled = depth_metric[vv, uu]
    r_sampled = frame_rgb[vv, uu, 0]
    g_sampled = frame_rgb[vv, uu, 1]
    b_sampled = frame_rgb[vv, uu, 2]
    
    uu_flat = uu.flatten().astype(np.float64)
    vv_flat = vv.flatten().astype(np.float64)
    z_flat = depth_sampled.flatten().astype(np.float64)
    r_flat = r_sampled.flatten()
    g_flat = g_sampled.flatten()
    b_flat = b_sampled.flatten()
    
    # Filter valid depth
    valid = (z_flat > MIN_DEPTH_M) & (z_flat < MAX_DEPTH_M)
    uu_flat = uu_flat[valid]
    z_flat = z_flat[valid]
    r_flat = r_flat[valid]
    g_flat = g_flat[valid]
    b_flat = b_flat[valid]
    
    # Compute X (lateral) from pixel position and depth
    X = (uu_flat - C_X) * z_flat / F_X
    Z = z_flat
    
    # FLATTEN: Force ALL road points to ground level (constant Y)
    Y = np.full_like(X, H_CAM)
    
    # Restrict lateral range to road area only (±8 meters)
    lateral_mask = np.abs(X) < 8.0
    X = X[lateral_mask]
    Y = Y[lateral_mask]
    Z = Z[lateral_mask]
    r_flat = r_flat[lateral_mask]
    g_flat = g_flat[lateral_mask]
    b_flat = b_flat[lateral_mask]
    
    points = np.stack([X, Y, Z], axis=1)
    colors = np.stack([r_flat, g_flat, b_flat], axis=1)
    
    return points, colors


# ─── Vehicle class dimensions (length, width, height) in meters ──────
VEHICLE_DIMS = {
    "car":        (4.0, 1.8, 1.5),
    "truck":      (8.0, 2.5, 3.0),
    "bus":        (10.0, 2.5, 3.2),
    "motorcycle": (2.0, 0.8, 1.2),
    "bicycle":    (1.8, 0.6, 1.1),
    "pedestrian": (0.5, 0.5, 1.7),
}


def compute_3d_bbox(x1, y1, x2, y2, depth_metric, class_name):
    """
    Compute a 3D bounding box (8 corners of a cuboid) from a 2D detection.
    
    Uses the median depth inside the bounding box as the object's Z distance,
    then back-projects the 2D corners into 3D and adds a class-based depth extent.
    
    Returns:
        corners: 8x3 array of (X, Y, Z) corner coordinates, or None if invalid
    """
    # Sample depth inside the bounding box (bottom 40% is most reliable)
    box_h = int(y2 - y1)
    box_w = int(x2 - x1)
    
    patch_top = max(0, int(y2 - box_h * 0.4))
    patch_bottom = min(FRAME_H - 1, int(y2))
    patch_left = max(0, int(x1))
    patch_right = min(FRAME_W - 1, int(x2))
    
    depth_patch = depth_metric[patch_top:patch_bottom, patch_left:patch_right]
    valid_depths = depth_patch[depth_patch > MIN_DEPTH_M]
    
    if len(valid_depths) == 0:
        return None
    
    Z_center = float(np.median(valid_depths))
    
    if Z_center < MIN_DEPTH_M or Z_center > MAX_DEPTH_M:
        return None
    
    # Get class-based dimensions
    length, width, height = VEHICLE_DIMS.get(class_name, (3.0, 1.5, 1.5))
    
    # Back-project 2D bbox center to 3D for X and Z
    cx_2d = (x1 + x2) / 2.0
    
    X_center = (cx_2d - C_X) * Z_center / F_X
    
    # Build 8 corners of the cuboid
    # Half-extents
    hw = width / 2.0   # half width (X axis)
    hl = length / 2.0  # half length (Z axis)
    
    # Force the bounding box to sit exactly ON the ground plane
    # Positive Y is down, so ground is H_CAM, top of car is H_CAM - height
    Y_bottom = H_CAM
    Y_top = H_CAM - height
    
    corners = np.array([
        [X_center - hw, Y_top, Z_center - hl],     # 0: front-left-top
        [X_center + hw, Y_top, Z_center - hl],     # 1: front-right-top
        [X_center + hw, Y_bottom, Z_center - hl],  # 2: front-right-bottom
        [X_center - hw, Y_bottom, Z_center - hl],  # 3: front-left-bottom
        [X_center - hw, Y_top, Z_center + hl],     # 4: back-left-top
        [X_center + hw, Y_top, Z_center + hl],     # 5: back-right-top
        [X_center + hw, Y_bottom, Z_center + hl],  # 6: back-right-bottom
        [X_center - hw, Y_bottom, Z_center + hl],  # 7: back-left-bottom
    ])
    
    return corners


def generate_wireframe_points(corners, color, points_per_edge=30):
    """
    Generate densely sampled points along the 12 edges of a cuboid.
    This creates a visible wireframe in the .ply point cloud.
    
    Args:
        corners: 8x3 array of cuboid corner positions
        color: (R, G, B) tuple for the wireframe color
        points_per_edge: number of points to sample per edge
    
    Returns:
        edge_points: Nx3 array of positions
        edge_colors: Nx3 array of RGB colors
    """
    # 12 edges of a cuboid (pairs of corner indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # back face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
    ]
    
    all_points = []
    all_colors = []
    
    for i, j in edges:
        p0 = corners[i]
        p1 = corners[j]
        # Linearly interpolate between corners
        for t in np.linspace(0, 1, points_per_edge):
            pt = p0 + t * (p1 - p0)
            all_points.append(pt)
            all_colors.append(color)
    
    return np.array(all_points), np.array(all_colors)


def main():
    # ─── Check video exists ──────────────────────────────────────────
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_PATH} not found.")
        return
    
    # ─── Create output directory ─────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ─── Load YOLO Model ─────────────────────────────────────────────
    print("Loading YOLOv8 model...")
    yolo_model = YOLO("yolov8n.pt")
    classes_of_interest = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
    class_names_map = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    
    # ─── Load MiDaS Depth Model ─────────────────────────────────────
    print("Loading MiDaS depth estimation model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    
    print(f"MiDaS loaded on {device}")
    
    # ─── Open Video ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open {VIDEO_PATH}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.1f} FPS")
    print(f"Saving a .ply every {FRAME_INTERVAL} frames ({FRAME_INTERVAL/fps:.1f}s intervals)")
    print(f"Downsampling: every {DOWNSAMPLE}th pixel")
    print(f"Depth range: {MIN_DEPTH_M}m – {MAX_DEPTH_M}m")
    print("-" * 60)
    
    # ─── Scale calibration (same approach as detect_and_hud.py) ──────
    # We use geometric depth: Z = (f_y * h_cam) / (v - v_horizon)
    # MiDaS gives relative inverse depth, so we calibrate a scale factor
    y_horizon = 310.0
    scale_factor = None
    scale_samples = []
    SCALE_WARMUP = 30
    
    frame_idx = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        frame = cv2.resize(frame_raw, (FRAME_W, FRAME_H))
        
        # Only process frames at the specified interval
        if frame_idx % FRAME_INTERVAL != 1 and frame_idx != 1:
            # Still run depth on warmup frames for scale calibration
            if frame_idx <= SCALE_WARMUP and scale_factor is None:
                pass  # fall through to process
            else:
                continue
        
        print(f"Processing frame {frame_idx}/{total_frames}...")
        
        # ─── Run MiDaS ──────────────────────────────────────────────
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)
        
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(FRAME_H, FRAME_W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        
        disparity_map = prediction.cpu().numpy()
        
        # Convert disparity to relative depth (inverse)
        depth_relative = np.zeros_like(disparity_map)
        valid_mask = disparity_map > 1e-3
        depth_relative[valid_mask] = 1.0 / disparity_map[valid_mask]
        
        # ─── Scale Calibration ───────────────────────────────────────
        # Use ground plane pixels to calibrate MiDaS scale
        if scale_factor is None:
            for v_sample in range(500, 700, 20):
                for u_sample in range(400, 880, 40):
                    midas_d = depth_relative[v_sample, u_sample]
                    if midas_d > 1e-6:
                        y_diff = max(v_sample - y_horizon, 3.0)
                        z_geo = (F_Y * H_CAM) / y_diff
                        if 2.0 < z_geo < 40.0:
                            scale_samples.append(z_geo / midas_d)
            
            if len(scale_samples) >= 20:
                scale_factor = float(np.median(scale_samples))
                print(f"  Scale factor calibrated: {scale_factor:.2f}")
        
        if scale_factor is None:
            print(f"  Skipping frame {frame_idx} (scale not yet calibrated)")
            continue
        
        # ─── Compute Metric Depth ────────────────────────────────────
        depth_metric = depth_relative * scale_factor
        
        # ─── Generate Point Cloud ────────────────────────────────────
        points, colors = depth_to_point_cloud(depth_metric, img_rgb, downsample=DOWNSAMPLE)
        
        if len(points) == 0:
            print(f"  No valid points for frame {frame_idx}, skipping.")
            continue
        
        # ─── Run YOLO Detection ──────────────────────────────────────
        results = yolo_model(frame, classes=classes_of_interest, verbose=False, imgsz=640)
        
        bbox_points_list = []
        bbox_colors_list = []
        det_count = 0
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cls = classes[i]
                conf = float(confidences[i])
                class_name = class_names_map.get(cls, "unknown")
                
                if conf < 0.4:
                    continue
                
                # Compute 3D bounding box
                corners = compute_3d_bbox(x1, y1, x2, y2, depth_metric, class_name)
                if corners is None:
                    continue
                
                # Choose wireframe color based on distance
                Z_obj = corners[:, 2].mean()
                if Z_obj < 5.0:
                    wire_color = (255, 0, 0)    # Red = close/danger
                elif Z_obj < 10.0:
                    wire_color = (255, 165, 0)  # Orange = medium
                else:
                    wire_color = (0, 100, 255)  # Blue = far/safe
                
                # Generate wireframe edge points
                edge_pts, edge_cols = generate_wireframe_points(corners, wire_color)
                bbox_points_list.append(edge_pts)
                bbox_colors_list.append(edge_cols)
                det_count += 1
        
        # ─── Generate Ground Plane Grid ───────────────────────────────
        grid_points, grid_colors = generate_ground_grid()
        
        # ─── Combine scene + bounding boxes + grid ───────────────────
        # User requested to remove the "unnecessary view" (the image point cloud)
        # and ONLY show the ground grid and the bounding boxes.
        parts = []
        color_parts = []
        
        if bbox_points_list:
            parts.append(np.vstack(bbox_points_list))
            color_parts.append(np.vstack(bbox_colors_list))
        
        parts.append(grid_points)
        color_parts.append(grid_colors)
        
        combined_points = np.vstack(parts)
        combined_colors = np.vstack(color_parts)
        
        pct = (frame_idx / total_frames) * 100
        print(f"  Detected {det_count} objects with 3D bounding boxes")
        print(f"  Progress: {pct:.0f}% ({frame_idx}/{total_frames})")
        
        # ─── Save .ply ───────────────────────────────────────────────
        ply_filename = f"frame_{frame_idx:06d}_boxes.ply"
        ply_path = os.path.join(OUTPUT_DIR, ply_filename)
        write_ply(ply_path, combined_points, combined_colors)
        saved_count += 1
    
    cap.release()
    print("-" * 60)
    print(f"Done! Saved {saved_count} point cloud files to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
