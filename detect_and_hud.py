import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

class EgoSpeedEstimator:
    def __init__(self, f_y, h_cam, y_horizon, fps):
        self.f_y = f_y
        self.h_cam = h_cam
        self.y_horizon = y_horizon
        self.fps = fps
        self.prev_gray = None
        self.prev_pts = None
        self.current_speed_kmh = 12.0  # Initial guess
        self.alpha_smooth = 0.05       # EMA smoothing factor
        
        # LK parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
    def update(self, frame, yolo_boxes):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Define Road ROI (relative to 1280x720)
        roi_y1 = int(self.y_horizon + 75)
        roi_y2 = int(height - 60)  # avoid car hood
        roi_x1 = int(width * 0.25)
        roi_x2 = int(width * 0.75)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self._detect_features(gray, roi_x1, roi_x2, roi_y1, roi_y2, yolo_boxes)
            return self.current_speed_kmh
            
        # Track features
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )
            
            if next_pts is not None:
                good_new = next_pts[status == 1]
                good_old = self.prev_pts[status == 1]
                
                speeds = []
                valid_new_pts = []
                for pt_new, pt_old in zip(good_new, good_old):
                    x_new, y_new = pt_new.ravel()
                    x_old, y_old = pt_old.ravel()
                    
                    dx = x_new - x_old
                    dy = y_new - y_old
                    
                    # We only care about points moving downwards (approaching us)
                    # and mostly straight (low horizontal shift compared to vertical)
                    if dy > 0.05 and abs(dx) < 1.0 * dy:
                        denom = (y_old - self.y_horizon) ** 2
                        if denom > 1.0:
                            v_m_s = (self.f_y * self.h_cam / denom) * dy * self.fps
                            v_kmh = v_m_s * 3.6
                            
                            # Sanity check: keep speeds between 0 and 150 km/h
                            if 0.0 <= v_kmh <= 150.0:
                                speeds.append(v_kmh)
                                valid_new_pts.append([x_new, y_new])
                
                if len(speeds) > 0:
                    inst_speed_kmh = np.median(speeds)
                    self.current_speed_kmh = (
                        self.alpha_smooth * inst_speed_kmh + 
                        (1 - self.alpha_smooth) * self.current_speed_kmh
                    )
                    self.prev_pts = np.array(valid_new_pts, dtype=np.float32).reshape(-1, 1, 2)
                else:
                    self.prev_pts = None
            else:
                self.prev_pts = None
        else:
            self.prev_pts = None
            
        # Re-detect if tracking lost or too few points
        if self.prev_pts is None or len(self.prev_pts) < 15:
            self._detect_features(gray, roi_x1, roi_x2, roi_y1, roi_y2, yolo_boxes)
            
        self.prev_gray = gray
        return self.current_speed_kmh
        
    def _detect_features(self, gray, x1, x2, y1, y2, yolo_boxes):
        # Create a mask for features
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        
        # Mask out any detected objects
        for box in yolo_boxes:
            bx1, by1, bx2, by2 = map(int, box[:4])
            margin = 10
            cv2.rectangle(
                mask, 
                (max(0, bx1 - margin), max(0, by1 - margin)), 
                (min(mask.shape[1] - 1, bx2 + margin), min(mask.shape[0] - 1, by2 + margin)), 
                0, 
                -1
            )
            
        # Find good features to track
        pts = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=50, 
            qualityLevel=0.01, 
            minDistance=15, 
            mask=mask
        )
        if pts is not None:
            self.prev_pts = pts.astype(np.float32)
        else:
            self.prev_pts = None

class LaneTracker:
    def __init__(self, width, height, y_horizon):
        self.width = width
        self.height = height
        self.y_horizon = y_horizon
        
        # Default lane lines for 1280x720 (confined strictly to the lower road surface):
        # Left line connects roughly (320, 720) and (480, 500) => x = a * y + b
        # Right line connects roughly (880, 720) and (730, 500) => x = a * y + b
        self.default_left = [-0.7273, 843.64]
        self.default_right = [0.6818, 389.10]
        
        self.left_fit = np.array(self.default_left)
        self.right_fit = np.array(self.default_right)
        self.alpha = 0.08  # strong smoothing for stable lines
        
        self.detected_consecutive_misses = 0
        
    def process(self, frame, yolo_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Mask representing our driving lane trapezoid (confined to the lower road surface)
        mask = np.zeros_like(gray)
        roi_pts = np.array([
            [320, 720],
            [470, 500],
            [730, 500],
            [880, 720]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [roi_pts], 255)
        
        # Mask out other vehicle bounding boxes to ignore edges of cars
        for box in yolo_boxes:
            bx1, by1, bx2, by2 = map(int, box[:4])
            cv2.rectangle(mask, (bx1, by1), (bx2, by2), 0, -1)
            
        # Detect white lane markings (high brightness)
        _, white_mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
        
        # Detect yellow lane markings (color HSV range)
        lower_yellow = np.array([12, 50, 90])
        upper_yellow = np.array([38, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine white/yellow markings, apply trapezoid ROI
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        lane_mask = cv2.bitwise_and(lane_mask, mask)
        
        # Get coordinates of positive pixels
        y_idx, x_idx = np.where(lane_mask > 0)
        
        left_x, left_y = [], []
        right_x, right_y = [], []
        
        # Split points based on center (640)
        for x, y in zip(x_idx, y_idx):
            if x < 630:
                left_x.append(x)
                left_y.append(y)
            elif x > 650:
                right_x.append(x)
                right_y.append(y)
                
        # Fit lines: x = a * y + b
        detected_left = False
        detected_right = False
        
        if len(left_x) > 30:
            left_fit_curr = np.polyfit(left_y, left_x, 1)
            # Verify slope is slanted correctly
            if -2.0 < left_fit_curr[0] < -0.3:
                self.left_fit = self.alpha * left_fit_curr + (1 - self.alpha) * self.left_fit
                detected_left = True
                
        if len(right_x) > 30:
            right_fit_curr = np.polyfit(right_y, right_x, 1)
            # Verify slope is slanted correctly
            if 0.3 < right_fit_curr[0] < 2.0:
                self.right_fit = self.alpha * right_fit_curr + (1 - self.alpha) * self.right_fit
                detected_right = True
                
        if not detected_left or not detected_right:
            self.detected_consecutive_misses += 1
            if self.detected_consecutive_misses > 15:
                # Decay slowly to default lines to keep dashboard clean
                self.left_fit = 0.05 * np.array(self.default_left) + 0.95 * self.left_fit
                self.right_fit = 0.05 * np.array(self.default_right) + 0.95 * self.right_fit
        else:
            self.detected_consecutive_misses = 0
            
        y_top = 500
        y_bot = 720
        
        x_left_top = int(self.left_fit[0] * y_top + self.left_fit[1])
        x_left_bot = int(self.left_fit[0] * y_bot + self.left_fit[1])
        x_right_top = int(self.right_fit[0] * y_top + self.right_fit[1])
        x_right_bot = int(self.right_fit[0] * y_bot + self.right_fit[1])
        
        return (x_left_top, x_left_bot), (x_right_top, x_right_bot)


class BEVPointCloudHUD:
    def __init__(self, width=240, height=240, scale_m=10.0, max_z=24.0, max_x=12.0):
        self.w = width
        self.h = height
        self.scale_m = scale_m  # pixels per meter (10 px = 1m)
        self.max_z = max_z
        self.max_x = max_x
        
    def to_bev_coords(self, X, Z):
        # Center of X is at self.w / 2
        # Bottom of Z is at self.h
        u = int(self.w / 2 + X * self.scale_m)
        v = int(self.h - Z * self.scale_m)
        return u, v
        
    def draw_base_grid(self):
        # Create true black background for BEVDriver look
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        canvas[:] = (0, 0, 0)
        
        # Draw Cartesian Grid
        for z_val in range(10, int(self.max_z) + 1, 10):
            _, v = self.to_bev_coords(0, z_val)
            cv2.line(canvas, (0, v), (self.w, v), (30, 30, 30), 1, cv2.LINE_4)
            cv2.putText(canvas, f"{int(z_val)}m", (5, v - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
            
        for x_val in range(-10, 11, 5):
            u, _ = self.to_bev_coords(x_val, 0)
            cv2.line(canvas, (u, 0), (u, self.h), (30, 30, 30), 1, cv2.LINE_4)
            if x_val != 0:
                cv2.putText(canvas, f"{x_val:+.0f}m", (u - 15, self.h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA)

        # Draw ego vehicle (solid white rectangle, BEVDriver style)
        ego_u, ego_v = self.to_bev_coords(0, 0)
        ego_w, ego_h = 10, 20
        cv2.rectangle(canvas, (ego_u - ego_w//2, ego_v - ego_h), (ego_u + ego_w//2, ego_v), (255, 255, 255), -1)

        # Draw Title
        cv2.putText(canvas, "Semantic BEV Map", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        return canvas

    def draw(self, vehicles, road_points=None):
        # Start with the base grid
        canvas = self.draw_base_grid()
        
        # Draw road points (semantic drivable area blob)
        if road_points is not None:
            xs, zs, colors = road_points
            for i, (x, z) in enumerate(zip(xs, zs)):
                if i % 2 == 0:
                    u, v = self.to_bev_coords(x, z)
                    if 0 <= u < self.w and 0 <= v < self.h:
                        # Dark purple semantic color for drivable area
                        cv2.circle(canvas, (u, v), 2, (70, 50, 50), -1)
        
        # Draw vehicles (semantic bounding boxes)
        for X, Z, color, track_id in vehicles:
            u, v = self.to_bev_coords(X, Z)
            
            # Skip if out of bounds
            if not (0 <= u < self.w and 0 <= v < self.h):
                continue
                
            # Draw vehicle marker (clean, smaller semantic rectangle)
            box_w = int(1.5 * self.scale_m) # 1.5m wide
            box_h = int(3.0 * self.scale_m) # 3.0m long
            
            pt1 = (u - box_w // 2, v - box_h)
            pt2 = (u + box_w // 2, v)
            
            # Solid semantic color block
            cv2.rectangle(canvas, pt1, pt2, color, -1)
            
            # Draw track ID label
            if track_id != -1:
                id_str = f"ID {track_id}"
                cv2.putText(canvas, id_str, (u - 15, v - box_h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Legend
        legend_y = self.h - 40
        cv2.rectangle(canvas, (10, legend_y-4), (18, legend_y+4), (0, 255, 0), -1)
        cv2.putText(canvas, "Safe", (25, legend_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.rectangle(canvas, (10, legend_y+12-4), (18, legend_y+12+4), (0, 255, 255), -1)
        cv2.putText(canvas, "Warn", (25, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.rectangle(canvas, (10, legend_y+24-4), (18, legend_y+24+4), (0, 0, 255), -1)
        cv2.putText(canvas, "Danger", (25, legend_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                            
        return canvas


def main():
    video_path = "ADAS.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found in current directory.")
        return

    # --- Load Models ---
    print("Loading ultralytics YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    
    print("Loading MiDaS depth estimation model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval()
    
    # Load MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Native display resolution
    width = 1280
    height = 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: Resizing to {width}x{height} @ {fps:.2f} FPS. Total frames: {total_frames}")
    print("Press 'q' in the window to exit the live video feed.")

    # Camera geometric parameters (scaled for 1280x720)
    h_cam = 1.40         # Camera height in meters
    f_y_geom = 1700.0    # Halved from 3400.0
    y_horizon = 310.0    # Halved from 620.0

    # Classes of interest
    classes_of_interest = [0, 1, 2, 3, 5, 7]
    class_names_map = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # Tracking states: {track_id: {z_prev, v_prev}}
    tracking_states = {}
    alpha = 0.25
    beta = 0.15
    dt = 1.0 / fps

    # Scale calibration
    scale_factor = None
    scale_samples = []
    SCALE_WARMUP_FRAMES = 15

    # Initialize Ego Speed Estimator
    speed_estimator = EgoSpeedEstimator(f_y_geom, h_cam, y_horizon, fps)

    # Initialize Lane Tracker and box memory
    lane_tracker = LaneTracker(width, height, y_horizon)
    prev_boxes = []
    
    # Initialize BEV Point Cloud HUD
    bev_hud = BEVPointCloudHUD()

    # Initialize depth cache
    depth_relative = np.zeros((height, width), dtype=np.float32)

    frame_idx = 0
    window_name = "ADAS Perception - Optimized HUD Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret:
            break
            
        frame_idx += 1

        # Resize immediately to display resolution to resolve blurriness and speed up processing
        frame = cv2.resize(frame_raw, (width, height))

        # --- 0. Detect Lane Lines & Draw Overlay ---
        (x_left_top, x_left_bot), (x_right_top, x_right_bot) = lane_tracker.process(frame, prev_boxes)
        
        # Draw lane boundary lines (Yellow left line, White right line)
        cv2.line(frame, (x_left_bot, 720), (x_left_top, 500), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x_right_bot, 720), (x_right_top, 500), (255, 255, 255), 2, cv2.LINE_AA)

        # --- 1. MiDaS Depth Estimation (Every 3rd frame for speedup) ---
        run_depth = (frame_idx == 1) or (frame_idx % 3 == 0)
        if run_depth:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
            
            disparity_map = prediction.cpu().numpy()
            depth_relative = np.zeros_like(disparity_map)
            valid_mask = disparity_map > 1e-3
            depth_relative[valid_mask] = 1.0 / disparity_map[valid_mask]

        # --- 2. YOLO Tracking (Runs on 1280x720 native resolution) ---
        results = model.track(frame, persist=True, classes=classes_of_interest, verbose=False, imgsz=640)
        
        detected_boxes_for_flow = []
        lead_vehicle_candidate = None
        min_lead_z = float('inf')
        bev_vehicles = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None

            detected_boxes_for_flow = boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cls = classes[i]
                conf = float(confidences[i])
                track_id = int(track_ids[i]) if track_ids is not None else -1
                
                class_name = class_names_map.get(cls, "unknown")
                
                # --- 3. Sample Cached Depth Map ---
                box_h = int(y2 - y1)
                box_w = int(x2 - x1)
                
                patch_top = max(0, int(y2 - box_h * 0.2))
                patch_bottom = min(height - 1, int(y2))
                patch_left = max(0, int((x1 + x2) / 2 - box_w * 0.2))
                patch_right = min(width - 1, int((x1 + x2) / 2 + box_w * 0.2))
                
                depth_patch = depth_relative[patch_top:patch_bottom, patch_left:patch_right]
                if depth_patch.size > 0:
                    midas_depth_raw = float(np.median(depth_patch[depth_patch > 0])) if np.any(depth_patch > 0) else 0.0
                else:
                    midas_depth_raw = 0.0

                # --- 4. Scale Calibration ---
                y_bottom = y2
                y_diff = max(y_bottom - y_horizon, 3.0)
                z_geometric = (f_y_geom * h_cam) / y_diff
                
                if midas_depth_raw > 1e-6 and 10.0 < z_geometric < 60.0 and frame_idx <= SCALE_WARMUP_FRAMES:
                    sample = z_geometric / midas_depth_raw
                    scale_samples.append(sample)
                
                if len(scale_samples) >= 3:
                    scale_factor = float(np.median(scale_samples))
                
                # --- 5. Compute Final Metric Distance ---
                if scale_factor is not None and midas_depth_raw > 1e-6:
                    z_metric = scale_factor * midas_depth_raw
                else:
                    z_metric = z_geometric

                # --- 6. Alpha-Beta Filter for Smoothing ---
                x_center = (x1 + x2) / 2.0
                X_world_raw = (x_center - 640.0) * z_metric / 1700.0

                if track_id != -1:
                    if track_id in tracking_states:
                        state = tracking_states[track_id]
                        z_prev = state['z']
                        v_prev = state['v']
                        x_prev = state.get('x', X_world_raw)
                        
                        z_pred = z_prev + v_prev * dt
                        res_z = z_metric - z_pred
                        z_filtered = z_pred + alpha * res_z
                        v_filtered = v_prev + (beta / dt) * res_z
                        
                        # Apply Exponential Moving Average (EMA) for smooth lateral movement
                        x_filtered = 0.85 * x_prev + 0.15 * X_world_raw
                        
                        tracking_states[track_id] = {'z': z_filtered, 'v': v_filtered, 'x': x_filtered}
                    else:
                        z_filtered = z_metric
                        v_filtered = 0.0
                        x_filtered = X_world_raw
                        tracking_states[track_id] = {'z': z_filtered, 'v': v_filtered, 'x': x_filtered}
                else:
                    z_filtered = z_metric
                    v_filtered = 0.0
                    x_filtered = X_world_raw

                # --- 7. Check if Lead Vehicle (Closest in Ego Lane) ---
                # Use actual detected lane boundaries at y2 (bottom of bounding box)
                left_x_bound = lane_tracker.left_fit[0] * y2 + lane_tracker.left_fit[1]
                right_x_bound = lane_tracker.right_fit[0] * y2 + lane_tracker.right_fit[1]
                
                # STRICT lead vehicle check (3 conditions must ALL be true):
                # 1. Pixel lane check: car center must be INSIDE lane lines (NO tolerance)
                in_lane_bounds = left_x_bound <= x_center <= right_x_bound
                
                # 2. Geometric check: car must be within +/- 1.2m of straight-ahead path
                is_straight_ahead = abs(x_filtered) < 1.2
                
                # 3. Frame-center check: car must be in the center region of the frame
                #    (rejects cars clearly on the far left/right of the image)
                frame_center_x = 640.0
                is_centered = abs(x_center - frame_center_x) < 180
                
                in_ego_lane = in_lane_bounds and is_straight_ahead and is_centered
                
                is_vehicle = class_name in ["car", "truck", "bus", "motorcycle"]
                if in_ego_lane and is_vehicle:
                    if z_filtered < min_lead_z:
                        min_lead_z = z_filtered
                        lead_vehicle_candidate = {
                            'id': track_id,
                            'distance': z_filtered,
                            'conf': conf,
                            'box': (x1, y1, x2, y2),
                            'class': class_name,
                            'speed_relative': v_filtered
                        }

                # --- 8. Lane Position Classification & Bounding Box Coloring ---
                is_vehicle = class_name in ["car", "truck", "bus", "motorcycle"]
                
                if is_vehicle and x_center < left_x_bound - 15:
                    color = (255, 255, 0)  # Cyan for oncoming/opposite
                    thickness = 1
                    zone_label = "ONCOMING"
                elif is_vehicle and x_center > right_x_bound + 15:
                    color = (180, 180, 180)  # Gray for adjacent
                    thickness = 1
                    zone_label = "ADJACENT"
                else:
                    if z_filtered < 3.0:
                        color = (0, 0, 255) if frame_idx % 8 < 4 else (0, 0, 180)
                        thickness = 2
                        zone_label = "EMERGENCY"
                    elif z_filtered < 4.0:
                        color = (0, 0, 255)  # Red for distance < 6m
                        thickness = 2
                        zone_label = "DANGER"
                    elif z_filtered < 5.0:
                        color = (0, 0, 255)  # Red for distance < 6m
                        thickness = 2
                        zone_label = "WARNING"
                    elif z_filtered < 6.0:
                        color = (0, 0, 255)  # Red for distance < 6m
                        thickness = 1
                        zone_label = "CAUTION"
                    else:
                        color = (0, 230, 118)
                        thickness = 1
                        zone_label = "SAFE"

                # Draw sharp 2D bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

                # Flashing outer box for emergency
                if z_filtered < 3.0:
                    pad = 4
                    flash_color = (255, 255, 255) if frame_idx % 8 < 4 else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1) - pad, int(y1) - pad), 
                                  (int(x2) + pad, int(y2) + pad), flash_color, 1)
                
                # Format label
                dist_str = f"{z_filtered:.1f}m"
                speed_kmh = v_filtered * 3.6
                speed_str = f" {speed_kmh:+.1f}km/h" if abs(v_filtered) > 0.5 else ""
                
                is_lead_match = (lead_vehicle_candidate is not None and track_id == lead_vehicle_candidate['id'])
                lead_badge = " [LEAD]" if is_lead_match else ""
                
                label = f"ID:{track_id} {class_name} ({dist_str}{speed_str}){lead_badge} [{zone_label}]"
                
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (int(x1), int(y1) - text_h - 6), (int(x1) + text_w + 6, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1) + 3, int(y1) - 3), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                # Store for BEV projection
                bev_vehicles.append((x_filtered, z_filtered, color, track_id))

        # --- 9. Estimate Ego Speed ---
        ego_speed_kmh = speed_estimator.update(frame, detected_boxes_for_flow)
        ego_speed_m_s = ego_speed_kmh / 3.6

        # Calculate safe distance (minimum 6m as requested)
        safe_dist = max(6.0, (ego_speed_m_s * 2.0) + 2.0)

        # --- 9.5. Draw Lead Vehicle Annotations (Part A1) ---
        if lead_vehicle_candidate is not None:
            lx1, ly1, lx2, ly2 = lead_vehicle_candidate['box']
            l_dist = lead_vehicle_candidate['distance']
            
            # Determine status color (Green if safe, Red if critical)
            is_safe = l_dist >= safe_dist
            status_color = (0, 230, 118) if is_safe else (0, 0, 255) # Green vs Red
            
            # 1. Draw vertical line from vehicle's bottom-center to bottom of screen
            x_mid = int((lx1 + lx2) / 2.0)
            y_bottom = int(ly2)
            cv2.line(frame, (x_mid, y_bottom), (x_mid, height), status_color, 2)
            
            # 2. Draw large distance number next to the line (semi-transparent or dark background)
            text_y = int(y_bottom + (height - y_bottom) * 0.4)
            dist_label = f"{l_dist:.1f}m"
            
            (t_w, t_h), _ = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x_mid + 10, text_y - t_h - 5), (x_mid + 10 + t_w + 10, text_y + 5), (15, 15, 15), -1)
            cv2.rectangle(frame, (x_mid + 10, text_y - t_h - 5), (x_mid + 10 + t_w + 10, text_y + 5), status_color, 1)
            cv2.putText(frame, dist_label, (x_mid + 15, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
            
            # 3. Label the vehicle itself as "LEAD VEHICLE" in bold yellow/green text
            lead_text = "LEAD VEHICLE"
            lead_text_color = (0, 255, 255) if not is_safe else (0, 230, 118) # Yellow vs Green
            
            (lt_w, lt_h), _ = cv2.getTextSize(lead_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            
            # We stack the "LEAD VEHICLE" label box on top of the standard label box
            std_label = f"ID:{lead_vehicle_candidate['id']} {lead_vehicle_candidate['class']} ({l_dist:.1f}m)"
            (std_w, std_h), _ = cv2.getTextSize(std_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_top_y = int(ly1) - std_h - 6
            
            # Draw lead vehicle text box above it
            cv2.rectangle(frame, (int(lx1), label_top_y - lt_h - 8), (int(lx1) + lt_w + 8, label_top_y), (15, 15, 15), -1)
            cv2.rectangle(frame, (int(lx1), label_top_y - lt_h - 8), (int(lx1) + lt_w + 8, label_top_y), lead_text_color, 1)
            cv2.putText(frame, lead_text, (int(lx1) + 4, label_top_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, lead_text_color, 2, cv2.LINE_AA)

        # --- 10. Draw Translucent "Following: SAFE" HUD Panel (Top-Left) ---
        hud_w, hud_h = 360, 200
        hud_x, hud_y = 20, 20
        
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 15, 15), -1)
        cv2.addWeighted(hud_overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (120, 120, 120), 1)

        # Content in HUD
        if lead_vehicle_candidate is not None:
            actual_dist = lead_vehicle_candidate['distance']
            lead_conf = int(lead_vehicle_candidate['conf'] * 100)
            
            is_safe = actual_dist >= safe_dist
            status_text = "Following: SAFE" if is_safe else "Following: CRITICAL"
            status_color = (0, 230, 118) if is_safe else (0, 0, 255)
            
            dist_text = f"Distance: {actual_dist:.1f}m"
            safe_text = f"Safe Dist: {safe_dist:.1f}m"
            conf_text = f"Conf: {lead_conf}%"
        else:
            status_text = "Following: NO TARGET"
            status_color = (180, 180, 180)
            dist_text = "Distance: --"
            safe_text = f"Safe Dist: {safe_dist:.1f}m"
            conf_text = "Conf: --"

        speed_text = f"Speed: {ego_speed_kmh:.1f} km/h (est)"

        # Draw Status Indicator block
        block_size = 14
        block_x = hud_x + 20
        block_y = hud_y + 25
        cv2.rectangle(frame, (block_x, block_y), (block_x + block_size, block_y + block_size), status_color, -1)
        cv2.rectangle(frame, (block_x, block_y), (block_x + block_size, block_y + block_size), (255, 255, 255), 1)

        # Draw Status Text
        cv2.putText(frame, status_text, (block_x + 25, block_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

        # Draw details inside HUD (crisp, smaller fonts for 720p display)
        detail_color = (255, 255, 255)
        font_style = cv2.FONT_HERSHEY_SIMPLEX
        font_sz = 0.5
        font_thick = 1
        
        cv2.putText(frame, dist_text, (hud_x + 20, hud_y + 70), font_style, font_sz, detail_color, font_thick, cv2.LINE_AA)
        cv2.putText(frame, safe_text, (hud_x + 20, hud_y + 100), font_style, font_sz, detail_color, font_thick, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (hud_x + 20, hud_y + 130), font_style, font_sz, detail_color, font_thick, cv2.LINE_AA)
        cv2.putText(frame, speed_text, (hud_x + 20, hud_y + 160), font_style, font_sz, detail_color, font_thick, cv2.LINE_AA)

        # --- 11. Scale calibration status and HUD indicators ---
        if scale_factor is not None:
            cal_text = f"Depth Calibrated (scale: {scale_factor:.2f})"
            cal_color = (0, 230, 118)
        else:
            cal_text = f"Calibrating depth... ({len(scale_samples)}/{SCALE_WARMUP_FRAMES})"
            cal_color = (0, 255, 255)
            
        cv2.putText(frame, cal_text, (hud_x, hud_y + hud_h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cal_color, 1, cv2.LINE_AA)

        # --- 11.5. Compute Road Point Cloud for BEV ---
        road_points = None
        # Generate grid points in the road ROI
        grid_y, grid_x = np.mgrid[480:700:10, 240:1040:10]
        road_mask = np.ones_like(grid_x, dtype=bool)
        
        # Mask out regions covered by detected vehicle bounding boxes
        for box in prev_boxes:
            bx1, by1, bx2, by2 = map(int, box[:4])
            x_min_idx = max(0, (bx1 - 240) // 10)
            x_max_idx = min(grid_x.shape[1], (bx2 - 240) // 10 + 1)
            y_min_idx = max(0, (by1 - 480) // 10)
            y_max_idx = min(grid_x.shape[0], (by2 - 480) // 10 + 1)
            if x_min_idx < x_max_idx and y_min_idx < y_max_idx:
                road_mask[y_min_idx:y_max_idx, x_min_idx:x_max_idx] = False
                
        valid_ys = grid_y[road_mask]
        valid_xs = grid_x[road_mask]
        
        if len(valid_ys) > 0:
            if scale_factor is not None:
                z_vals = depth_relative[valid_ys, valid_xs] * scale_factor
            else:
                z_vals = (f_y_geom * h_cam) / np.maximum(valid_ys - y_horizon, 3.0)
                
            valid_depth_mask = (z_vals > 1.0) & (z_vals <= 24.0)
            z_filtered_vals = z_vals[valid_depth_mask]
            xs_filtered = valid_xs[valid_depth_mask]
            ys_filtered = valid_ys[valid_depth_mask]
            
            if len(z_filtered_vals) > 0:
                x_vals = (xs_filtered - 640.0) * z_filtered_vals / 1700.0
                colors = frame[ys_filtered, xs_filtered]
                road_points = (x_vals, z_filtered_vals, colors)

        # --- 12. Draw and Overlay BEV Point Cloud HUD (Top-Right) ---
        bev_canvas = bev_hud.draw(bev_vehicles, road_points)
        
        # Blend BEV canvas into the top-right corner of the frame
        bev_w, bev_h = bev_hud.w, bev_hud.h
        bev_x, bev_y = width - bev_w - 20, 20
        
        bev_roi = frame[bev_y:bev_y+bev_h, bev_x:bev_x+bev_w]
        blended_bev = cv2.addWeighted(bev_canvas, 0.75, bev_roi, 0.25, 0)
        frame[bev_y:bev_y+bev_h, bev_x:bev_x+bev_w] = blended_bev
        cv2.rectangle(frame, (bev_x, bev_y), (bev_x + bev_w, bev_y + bev_h), (120, 120, 120), 1)

        # Update prev_boxes for the next frame's lane masking
        prev_boxes = boxes if (results and results[0].boxes is not None) else []

        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit requested by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live video stream closed.")

if __name__ == "__main__":
    main()
