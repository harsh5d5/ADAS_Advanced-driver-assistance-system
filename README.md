# Real-Time ADAS Perception Pipeline from Fleet Dashcam 🚗

Real-world dashcam footage from fleet platforms like Motive provides rich data for developing Advanced Driver Assistance Systems (ADAS). This project implements a real-time ADAS perception pipeline that combines state-of-the-art deep learning, metric depth estimation, and 3D spatial projection to create an advanced driver-assistance experience.

---

## 🚀 Features

* **Multi-Object Detection & Tracking:** Real-time tracking of vehicles, pedestrians, and cycles using YOLOv8 and ByteTrack (or similar robust trackers).
* **Metric Depth Estimation & Lane Detection:** Deep-learning based relative depth maps calibrated to metric distance ($Z$) alongside advanced lane fitting and tracking.
* **Stabilized 3D Bounding Boxes:** 3D wireframe cuboid mapping of vehicles with dynamic distance and relative speed metrics.
* **Bird’s-Eye-View (BEV) Pseudo-LiDAR Visualization:** Clean, flat bird's-eye-view pseudo-LiDAR projection sitting flush on a metric Cartesian ground grid.
* **Collision Risk Assessment:** Adaptive risk calculation, color-coding safety thresholds to warn the driver instantly.
* **Cut-In Detection & Lead Vehicle Analysis:** Lock onto the closest preceding vehicle in the ego-lane with an integrated visual status line and dynamic distance monitoring.

---

## 🛠️ System Architecture

The pipeline consists of three core components:

1. **2D ADAS HUD (`detect_and_hud.py`):**
   * Real-time dashboard view showing bounding boxes, lane lines, ego-speed estimation, and a HUD panel.
   * Prominent **SAFE GO** / **CRITICAL WARNING** banner alerts.
   * Safety line linking your vehicle to the lead vehicle (turns **red** if closer than 6m, **green** if safe).
   
2. **3D Projection & Generation (`point_cloud_3d.py`):**
   * Generates flat grid-based point clouds and projects 3D vehicle cuboids based on depth estimation and 2D bounding boxes.
   * Outputs structured binary `.ply` files to `output/pointclouds/`.

3. **Interactive 3D Point Cloud Viewer (`view_pointcloud.py`):**
   * High-performance 3D visualizer built using Open3D.
   * Custom camera views to step through processed frames interactively.

---

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd ADAS
   ```

2. **Install the required dependencies:**
   Make sure you are using Python 3.10+ (System Python 3.12 recommended for Open3D support on Windows):
   ```bash
   pip install numpy opencv-python ultralytics torch open3d
   ```

3. **Make sure your video file `ADAS.mp4` is placed in the project root.**

---

## 🏃 How to Run

### 1. Run the 2D ADAS HUD (Real-Time Video Feed)
To view the live dashboard video stream with bounding boxes, lane overlays, speed estimation, and safety banners:
```bash
python detect_and_hud.py
```
*Press **q** to close the window.*

### 2. Generate the 3D Point Clouds
To process the video frames and output the 3D spatial grids and bounding boxes:
```bash
python point_cloud_3d.py
```

### 3. Run the Interactive 3D Viewer
To explore the generated 3D environments frame-by-frame:
```bash
python view_pointcloud.py
```

#### **3D Viewer Controls:**
* **D** or **Right Arrow:** Next frame
* **A** or **Left Arrow:** Previous frame
* **Mouse Drag:** Rotate camera view
* **Scroll Wheel:** Zoom in / out
* **Q:** Quit viewer

---

## 🎯 Key Use Cases

* **Real-time driver safety monitoring and event detection** – Warnings for forward collisions, tailgating, and lane departures.
* **Automated incident review and root cause analysis** – Clear visual reconstruction of commercial fleet driving incidents.
* **Training data generation for machine learning models** – Creating labeled 3D boxes and flat pseudo-LiDAR points from 2D camera footage.
* **Development of aftermarket ADAS solutions** – Cost-effective ADAS implementations using standard fleet dashcams.
* **Simulation and scenario replay** – Validating motion planning models in a synthetic 3D space.
* **Fleet risk assessment** – Analytics based on tailgating durations, close cut-ins, and driver warnings.
