"""
3D Point Cloud Viewer - Interactive Frame Navigation
=====================================================
Opens .ply files from output/pointclouds/ in an interactive 3D viewer.
Use Right/Left arrow keys to navigate between frames INSIDE the viewer.

Controls:
  - Left click + drag  -> Rotate
  - Scroll wheel       -> Zoom  
  - Middle click + drag -> Pan
  - Right Arrow (D)     -> Next frame
  - Left Arrow (A)      -> Previous frame
  - R                   -> Reset view
  - Q / Esc             -> Close
"""

import os
import sys
import glob
# pyrefly: ignore [missing-import]
import open3d as o3d
import numpy as np


class PointCloudNavigator:
    def __init__(self, ply_files):
        self.files = ply_files
        self.current_idx = 0
        self.vis = None
        self.pcd = None
        
    def load_frame(self, idx):
        """Load a point cloud frame by index."""
        self.current_idx = max(0, min(idx, len(self.files) - 1))
        filepath = self.files[self.current_idx]
        
        new_pcd = o3d.io.read_point_cloud(filepath)
        n_points = len(new_pcd.points)
        
        print(f"[{self.current_idx + 1}/{len(self.files)}] {os.path.basename(filepath)} - {n_points:,} points")
        return new_pcd
    
    def next_frame(self, vis):
        """Callback for next frame."""
        if self.current_idx < len(self.files) - 1:
            new_pcd = self.load_frame(self.current_idx + 1)
            self.pcd.points = new_pcd.points
            self.pcd.colors = new_pcd.colors
            vis.update_geometry(self.pcd)
        else:
            print("  (Already at last frame)")
        return False
    
    def prev_frame(self, vis):
        """Callback for previous frame."""
        if self.current_idx > 0:
            new_pcd = self.load_frame(self.current_idx - 1)
            self.pcd.points = new_pcd.points
            self.pcd.colors = new_pcd.colors
            vis.update_geometry(self.pcd)
        else:
            print("  (Already at first frame)")
        return False
    
    def run(self, start_idx=0):
        """Launch the interactive viewer."""
        # Load initial frame
        self.pcd = self.load_frame(start_idx)
        
        # Create viewer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="3D Point Cloud Viewer - Press D/Right=Next, A/Left=Prev, Q=Quit",
            width=1280, height=720
        )
        self.vis.add_geometry(self.pcd)
        
        # Register key callbacks
        # D key (68) and Right arrow (262) -> next frame
        self.vis.register_key_callback(68, self.next_frame)   # D
        self.vis.register_key_callback(262, self.next_frame)  # Right arrow
        
        # A key (65) and Left arrow (263) -> previous frame
        self.vis.register_key_callback(65, self.prev_frame)   # A
        self.vis.register_key_callback(263, self.prev_frame)  # Left arrow
        
        # Set render options
        opt = self.vis.get_render_option()
        opt.background_color = [0.05, 0.05, 0.05]
        opt.point_size = 2.5
        
        # Fix 3: Set initial camera to top-down angled view (like reference image)
        ctr = self.vis.get_view_control()
        ctr.set_front([0, -0.8, -0.6])    # Looking down at ~55 degree angle
        ctr.set_lookat([0, 1.0, 12])      # Focus on road 12m ahead
        ctr.set_up([0, -1, 0])            # Y-down
        ctr.set_zoom(0.3)
        
        print("\n--- 3D Point Cloud Viewer ---")
        print(f"Total frames: {len(self.files)}")
        print("Controls:")
        print("  D / Right Arrow -> Next frame")
        print("  A / Left Arrow  -> Previous frame")
        print("  Mouse drag      -> Rotate")
        print("  Scroll           -> Zoom")
        print("  Q                -> Quit")
        print("-" * 40)
        
        self.vis.run()
        self.vis.destroy_window()


def main():
    ply_dir = "output/pointclouds"
    
    # Check if a specific file was passed
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target) and target.endswith('.ply'):
            # Single file mode
            nav = PointCloudNavigator([target])
            nav.run()
            return
        elif os.path.isdir(target):
            ply_dir = target
    
    # Find all .ply files
    files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
    
    if not files:
        print(f"No .ply files found in '{ply_dir}'")
        print("Run point_cloud_3d.py first to generate .ply files.")
        return
    
    print(f"Found {len(files)} .ply files in '{ply_dir}'")
    
    # Launch interactive navigator
    nav = PointCloudNavigator(files)
    nav.run()


if __name__ == "__main__":
    main()
