import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

class SpaceAssessmentSystem:
    RELEVANT_CLASSES = [
        0,   # person
        56,  # chair
        57,  # couch
        58,  # potted plant
        59,  # bed
        60,  # dining table
        61,  # toilet
        62,  # tv
        63,  # laptop
        24,  # backpack
        26,  # handbag
        28,  # suitcase
        # You can add more classes if interested. Just make sure to look up the class IDs for COCO dataset
    ]
    
    def __init__(self, model_size='n', use_stereo=True, jetson_optimization=True):
        """
        Initialize the space assessment system
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            use_stereo: Whether to use stereo cameras for depth estimation
            jetson_optimization: Whether to apply Jetson-specific optimizations
        """
        # Load YOLOv8 model
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # Set device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if jetson_optimization and self.device.type == 'cuda':
            # For Jetson, we'll use specific optimizations later
            print("Jetson optimizations enabled")
            
        # Stereo camera parameters
        self.use_stereo = use_stereo
        if use_stereo:
            # Initialize stereo algorithm
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=128,
                blockSize=5,
                P1=8 * 3 * 5**2,
                P2=32 * 3 * 5**2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=32
            )
            
            # Initialize calibration parameters
            self.is_calibrated = False
            self.camera_matrix_left = None
            self.camera_matrix_right = None
            self.dist_coeffs_left = None
            self.dist_coeffs_right = None
            self.R = None
            self.T = None
            self.map1_left = None
            self.map2_left = None
            self.map1_right = None
            self.map2_right = None
            
        # Inference settings
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Floor plane parameters (to be estimated)
        self.floor_normal = None
        self.floor_point = None
        
        # Performance metrics
        self.inference_times = []
        
    def calibrate_stereo_cameras(self, left_images, right_images, chessboard_size=(9, 6)):
        """Calibrate stereo cameras using chessboard images"""
        if not self.use_stereo:
            return
        
        # Find chessboard corners
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        objpoints = []  # 3D points in real world space
        imgpoints_left = []  # 2D points in left image plane
        imgpoints_right = []  # 2D points in right image plane
        
        for left_img, right_img in zip(left_images, right_images):
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
            
            if ret_left and ret_right:
                objpoints.append(objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
                
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
        
        # Calibrate each camera individually
        ret_left, self.camera_matrix_left, self.dist_coeffs_left, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
        
        ret_right, self.camera_matrix_right, self.dist_coeffs_right, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_right, gray_right.shape[::-1], None, None)
        
        # Stereo calibration
        ret, _, _, _, _, self.R, self.T, _, _ = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            gray_left.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # Compute rectification parameters
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            gray_left.shape[::-1], self.R, self.T
        )
        
        # Compute mapping for rectification
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, R1, P1,
            gray_left.shape[::-1], cv2.CV_32FC1
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, R2, P2,
            gray_right.shape[::-1], cv2.CV_32FC1
        )
        
        self.is_calibrated = True
        print("Stereo calibration completed.")
    
    def compute_depth_map(self, left_img, right_img):
        """Compute depth map from stereo images"""
        if not self.use_stereo:
            return None
            
        # Check if calibration has been performed
        if not self.is_calibrated:
            # If not calibrated, use images as-is (will be less accurate)
            print("WARNING: Using uncalibrated stereo. Results will be less accurate.")
            
            # Convert to grayscale
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Compute disparity map
            disparity = self.stereo.compute(gray_left, gray_right)
            disparity = disparity.astype(np.float32) / 16.0
            
            # Since we don't have calibration, use a simple depth approximation
            # This is less accurate than properly calibrated stereo
            depth = np.zeros_like(disparity, dtype=np.float32)
            valid_mask = (disparity > 0)
            # Approximate depth: baseline * focal_length / disparity
            # Using approximate values
            baseline = 0.1  # 10cm between cameras (assumed)
            focal_length = 500  # approximate focal length in pixels
            depth[valid_mask] = baseline * focal_length / disparity[valid_mask]
            
            return depth
        else:
            # Use calibrated stereo
            # Rectify images
            rect_left = cv2.remap(left_img, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
            rect_right = cv2.remap(right_img, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
            
            # Convert to grayscale
            gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
            
            # Compute disparity map
            disparity = self.stereo.compute(gray_left, gray_right)
            disparity = disparity.astype(np.float32) / 16.0
            
            # Filter out invalid values
            valid_mask = (disparity > 0)
            
            # Convert disparity to depth using calibration information
            depth = np.zeros_like(disparity, dtype=np.float32)
            depth[valid_mask] = self.T[0] * self.camera_matrix_left[0, 0] / disparity[valid_mask]
            
            return depth
    
    def estimate_floor_plane(self, depth_map, left_img):
        """Estimate floor plane from depth map using RANSAC"""
        if not self.use_stereo or depth_map is None:
            # If no stereo, assume floor is at y=0 plane
            self.floor_normal = np.array([0, 1, 0])
            self.floor_point = np.array([0, 0, 0])
            return
            
        # Create point cloud from depth map
        height, width = depth_map.shape
        points = []
        
        for y in range(0, height, 10):  # Sample every 10 pixels for efficiency
            for x in range(0, width, 10):
                if depth_map[y, x] > 0:
                    # Convert to 3D point
                    point = np.array([x, y, 1.0]) * depth_map[y, x]
                    points.append(point)
        
        points = np.array(points)
        
        # Use RANSAC to find floor plane
        if len(points) > 100:  # Need enough points for stable estimation
            max_inliers = 0
            best_normal = None
            best_point = None
            
            for _ in range(100):  # Number of RANSAC iterations
                # Randomly select 3 points
                indices = np.random.choice(len(points), 3, replace=False)
                p1, p2, p3 = points[indices]
                
                # Compute plane normal
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                normal = normal / np.linalg.norm(normal)
                
                # Count inliers
                dists = np.abs(np.dot(points - p1, normal))
                inliers = np.sum(dists < 0.1)  # Threshold for inliers
                
                if inliers > max_inliers:
                    max_inliers = inliers
                    best_normal = normal
                    best_point = p1
            
            if best_normal is not None:
                # Ensure normal points upward
                if best_normal[1] < 0:
                    best_normal = -best_normal
                
                self.floor_normal = best_normal
                self.floor_point = best_point
                
                print(f"Floor plane estimated with {max_inliers} inliers.")
    
    def detect_objects(self, frame):
        """Detect objects in a single frame"""
        start_time = time.time()
        
        # Inference with YOLOv8
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, 
                            classes=self.RELEVANT_CLASSES)
        
        self.inference_times.append(time.time() - start_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        # Extract detections
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls.item())
                if class_id in self.RELEVANT_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf.item()
                    category = self.model.names[class_id]
                    
                    detections.append({
                        'class_id': class_id,
                        'category': category,
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence
                    })
        
        return detections
    
    def process_stereo_frames(self, left_frame, right_frame):
        """Process stereo frames to detect objects and estimate occupied space"""
        # Compute depth map
        depth_map = None
        if self.use_stereo:
            depth_map = self.compute_depth_map(left_frame, right_frame)
            
            # Estimate floor plane if not estimated yet
            if self.floor_normal is None:
                self.estimate_floor_plane(depth_map, left_frame)
        
        # Detect objects in left frame
        detections = self.detect_objects(left_frame)
        
        # Calculate occupied areas
        occupied_area = self.calculate_occupied_area(detections, depth_map, left_frame.shape[1], left_frame.shape[0])
        
        # Calculate average FPS
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return detections, occupied_area, fps
    
    def calculate_occupied_area(self, detections, depth_map, frame_width, frame_height):
        """Calculate occupied area from detections and depth map"""
        # Create an occupancy map (top-down view)
        occupancy_map = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            category = detection['category']
            
            # Different approach based on whether we have depth information
            if self.use_stereo and depth_map is not None:
                # Extract depth for this bounding box
                box_depth = depth_map[y1:y2, x1:x2]
                if box_depth.size > 0:
                    # Calculate mean valid depth
                    valid_depths = box_depth[box_depth > 0]
                    if valid_depths.size > 0:
                        mean_depth = np.mean(valid_depths)
                        
                        # Project to floor plane
                        box_center_x = (x1 + x2) // 2
                        box_center_y = y2  # Use bottom of bounding box as contact point with floor
                        
                        # Calculate 3D position
                        if self.floor_normal is not None:
                            # Simple projection to floor for demonstration
                            # A more accurate approach would use camera parameters
                            floor_x = box_center_x
                            floor_z = mean_depth
                            
                            # Estimate object footprint based on category
                            if category == 'person':
                                radius = int(30 * 100 / mean_depth)  # Scale with depth
                            elif 'chair' in category or 'bag' in category:
                                radius = int(20 * 100 / mean_depth)
                            else:  # Furniture
                                radius = int(50 * 100 / mean_depth)
                            
                            # Mark occupancy (limit radius to reasonable values)
                            radius = min(max(radius, 5), 100)
                            cv2.circle(occupancy_map, (floor_x, int(floor_z)), radius, 255, -1)
            else:
                # Without depth, use simple heuristics based on position in image
                # Assume lower in image = closer to camera
                width = x2 - x1
                height = y2 - y1
                position_factor = y2 / frame_height  # 0 at top, 1 at bottom
                
                # Estimate object footprint based on category and position
                if category == 'person':
                    radius = int(width * 0.7)
                elif 'chair' in category or 'bag' in category:
                    radius = int(width * 0.8)
                else:  # Furniture
                    radius = int(width * 1.2)
                    
                # Adjust radius based on position (closer objects appear larger)
                radius = int(radius * (0.5 + 0.5 * position_factor))
                
                # Mark occupancy
                center_x = (x1 + x2) // 2
                center_y = y2  # Bottom of object
                cv2.circle(occupancy_map, (center_x, center_y), radius, 255, -1)
        
        # Calculate occupied area percentage
        total_pixels = frame_width * frame_height
        occupied_pixels = np.sum(occupancy_map > 0)
        occupied_percentage = 100.0 * occupied_pixels / total_pixels
        
        return {
            'occupancy_map': occupancy_map,
            'occupied_pixels': occupied_pixels,
            'total_pixels': total_pixels,
            'occupied_percentage': occupied_percentage
        }
    
    def visualize_results(self, left_frame, detections, occupied_area):
        """Visualize detection results and occupied area"""
        # Create a copy of the frame for visualization
        vis_frame = left_frame.copy()
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            category = detection['category']
            confidence = detection['confidence']
            
            # Different color based on category
            if category == 'person':
                color = (0, 255, 0)  # Green
            elif 'chair' in category or 'couch' in category or 'bed' in category or 'table' in category:
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 0, 255)  # Red
                
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{category} {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw occupancy information
        occupied_percentage = occupied_area['occupied_percentage']
        cv2.putText(vis_frame, f"Occupied: {occupied_percentage:.1f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Draw occupancy map (scaled down)
        occupancy_map = occupied_area['occupancy_map']
        h, w = occupancy_map.shape
        occupancy_vis = cv2.resize(occupancy_map, (w//4, h//4))
        occupancy_vis_color = cv2.cvtColor(occupancy_vis, cv2.COLOR_GRAY2BGR)
        
        # Place occupancy map in corner
        roi_h, roi_w = occupancy_vis.shape
        vis_frame[10:10+roi_h, 10:10+roi_w, :] = occupancy_vis_color
        
        return vis_frame

def simulate_stereo():
    """Simulate stereo camera input for testing purposes"""
    # Create a synthetic left and right image pair
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some objects in the images
    # Person
    cv2.rectangle(left, (100, 100), (200, 400), (0, 0, 255), -1)
    cv2.rectangle(right, (90, 100), (190, 400), (0, 0, 255), -1)  # Shifted slightly
    
    # Chair
    cv2.rectangle(left, (300, 200), (400, 400), (0, 255, 0), -1)
    cv2.rectangle(right, (290, 200), (390, 400), (0, 255, 0), -1)  # Shifted slightly
    
    # Table
    cv2.rectangle(left, (450, 300), (600, 400), (255, 0, 0), -1)
    cv2.rectangle(right, (440, 300), (590, 400), (255, 0, 0), -1)  # Shifted slightly
    
    return left, right

def process_webcam():
    """Process video from webcam or video file"""
    # Initialize system
    system = SpaceAssessmentSystem(model_size='n', use_stereo=False, jetson_optimization=True)
    
    # Open video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or a video file path
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # For single camera setup (no stereo)
        detections = system.detect_objects(frame)
        occupied_area = system.calculate_occupied_area(
            detections, None, frame.shape[1], frame.shape[0])
        
        # Visualize results
        vis_frame = system.visualize_results(frame, detections, occupied_area)
        
        # Display the frame
        cv2.imshow('YOLOv8 Space Assessment', vis_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_stereo_cameras():
    """Process video from stereo cameras"""
    # Initialize system
    system = SpaceAssessmentSystem(model_size='n', use_stereo=True, jetson_optimization=True)
    
    # Simulate continuous processing
    while True:
        # Get frames from stereo cameras (simulated here)
        left_frame, right_frame = simulate_stereo()
        
        # Process frames
        detections, occupied_area, fps = system.process_stereo_frames(left_frame, right_frame)
        
        # Visualize results
        vis_frame = system.visualize_results(left_frame, detections, occupied_area)
        
        # Add FPS counter
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (vis_frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('YOLOv8 Space Assessment (Stereo)', vis_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def main():
    """Main function to select processing mode"""
    print("YOLOv8 Space Assessment System")
    print("1. Process webcam feed (no stereo)")
    print("2. Process stereo cameras (simulated)")
    
    choice = input("Select mode (1/2): ")
    
    if choice == '1':
        process_webcam()
    elif choice == '2':
        process_stereo_cameras()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()