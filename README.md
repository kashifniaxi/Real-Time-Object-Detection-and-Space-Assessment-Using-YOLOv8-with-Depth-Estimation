# "Real-Time Object Detection and Space Assessment Using YOLOv8 with Depth Estimation and Jetson Nano Optimization"

## Introduction
This project utilizes YOLOv8 for detecting people and various objects (furniture, baby carriers, and bags) in an indoor environment. The goal is to analyze the occupied space and calculate the available area. The system integrates depth estimation using stereo cameras and optimizations for real-time processing on Jetson Nano.

## Features
- **Object Detection:** Uses YOLOv8 to identify objects in real-time.
- **Depth Estimation:** Implements stereo vision for spatial analysis.
- **Occupied Space Calculation:** Estimates the area occupied by detected objects.
- **Jetson Nano Deployment:** Optimized for efficient inference on edge devices.

## System Requirements
### Hardware:
- NVIDIA Jetson Nano (Recommended for deployment)
- CUDA-enabled GPU (For training and testing)
- Stereo cameras (For depth estimation)

### Software Dependencies:
Ensure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```
#### Required Libraries:
- Python 3.x
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- NumPy
- Streamlit (for UI)
- Torchvision

## Installation and Setup
1. Clone the repository:
```bash
git clone https://github.com/your-repo.git
cd your-repo
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download YOLOv8 model:
```bash
wget https://github.com/ultralytics/assets/releases/download/v8/yolov8n.pt
```
4. Run the application:
```bash
streamlit run app.py
```

## Usage
### Running Object Detection
- **Webcam Mode:** Runs YOLOv8 on live camera feed.
```bash
python Obj_Detection_Yolov8.py
```
- **Stereo Camera Mode:** Runs detection with depth estimation.
```bash
python Obj_Detection_Yolov8.py --stereo
```

### Deployment on Jetson Nano
- Convert the model for TensorRT optimization:
```bash
python export.py --weights yolov8n.pt --include engine
```
- Run optimized inference:
```bash
python Obj_Detection_Yolov8.py --jetson
```

## System Workflow
1. **Object Detection:** YOLOv8 detects objects in real-time.
2. **Depth Estimation:** If stereo cameras are used, depth maps are generated.
3. **Occupied Space Calculation:** The area occupied by objects is estimated.
4. **Visualization:** Bounding boxes and heatmaps overlay the detected objects and space usage.

## Performance Optimization
- **Jetson Nano Optimization:** Uses TensorRT for efficient inference.
- **Batch Processing:** Reduces computational load.
- **Threshold Tuning:** Adjusts confidence and IoU for improved accuracy.

## Future Improvements
- Integration with LiDAR for better depth estimation.
- Real-time spatial analytics for navigation assistance.
- Custom dataset training for specialized environments.


## License
This project is licensed under the MIT License.
