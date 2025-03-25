#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2

class RealSenseHandler:
    def __init__(self):
        rospy.init_node('realsense_d435_handler', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create subscribers
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.points_sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self.pointcloud_callback)
        
        # Store the latest frames
        self.latest_color_frame = None
        self.latest_depth_frame = None
        
        # Create publishers for processed images
        self.processed_color_pub = rospy.Publisher('/camera/processed/color', Image, queue_size=10)
        self.processed_depth_pub = rospy.Publisher('/camera/processed/depth', Image, queue_size=10)
        
        # Detection parameters (for student implementations)
        self.min_object_size = 1000  # Minimum object size in pixels
        self.distance_threshold = 0.1  # Distance threshold in meters
        
        rospy.loginfo("RealSense D435 handler initialized")
        
    def color_callback(self, data):
        """Callback for color images"""
        try:
            # Convert ROS Image message to OpenCV image
            color_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.latest_color_frame = color_frame
            
            # Display the color image
            cv2.imshow("Color Image", color_frame)
            cv2.waitKey(1)
            
            # Call student implementation functions
            self.process_color_image(color_frame)
            
            # If we have both color and depth, call the combined processing function
            if self.latest_depth_frame is not None:
                self.process_rgbd_data(self.latest_color_frame, self.latest_depth_frame)
            
        except Exception as e:
            rospy.logerr("Error processing color image: {}".format(e))
    
    def depth_callback(self, data):
        """Callback for depth images"""
        try:
            # Convert ROS Image message to OpenCV image
            depth_frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.latest_depth_frame = depth_frame
            
            # Normalize the depth image for visualization
            depth_colormap = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
            
            # Display the depth image
            cv2.imshow("Depth Image", depth_colormap)
            cv2.waitKey(1)
            
            # Call student implementation functions
            self.process_depth_image(depth_frame, depth_colormap)
            
        except Exception as e:
            rospy.logerr("Error processing depth image: {}".format(e))
    
    def pointcloud_callback(self, data):
        """Callback for point cloud data"""
        try:
            # Basic point cloud reading
            gen = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
            point = next(gen)
            rospy.loginfo("Point from cloud: x={:.2f}, y={:.2f}, z={:.2f}".format(
                point[0], point[1], point[2]))
            
            # Call student implementation function
            self.process_pointcloud(data)
            
        except Exception as e:
            rospy.logerr("Error processing point cloud: {}".format(e))
    
    # ============ STUDENT IMPLEMENTATION FUNCTIONS ============
    
    def process_color_image(self, color_frame):
        """
        STUDENT TASK: Process the RGB color image from the camera
        
        Learn about:
        - Basic image processing techniques
        - Color space transformations
        - Feature detection in color images
        
        Parameters:
        - color_frame: The RGB image from the camera (OpenCV format)
        
        Implementation ideas:
        - Convert to different color spaces (HSV, Lab)
        - Apply filters (Gaussian, median)
        - Detect edges using Canny or Sobel
        - Implement basic object detection using color segmentation
        """
        # TODO: Implement this function
        
        # HINT: Try converting to different color spaces
        # hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        
        # HINT: Try applying filters
        # blurred = cv2.GaussianBlur(color_frame, (5, 5), 0)
        
        # HINT: Try edge detection
        # edges = cv2.Canny(blurred, 50, 150)
        
        # For now, just publish the original image
        if color_frame is not None:
            try:
                msg = self.bridge.cv2_to_imgmsg(color_frame, "bgr8")
                self.processed_color_pub.publish(msg)
            except Exception as e:
                rospy.logerr("Error publishing processed color image: {}".format(e))
    
    def process_depth_image(self, depth_frame, depth_colormap):
        """
        STUDENT TASK: Process the depth image from the camera
        
        Learn about:
        - Depth image interpretation
        - Distance measurements
        - Depth filtering and noise removal
        - Depth-based segmentation
        
        Parameters:
        - depth_frame: The raw depth image (distances in mm)
        - depth_colormap: The colorized version for visualization
        
        Implementation ideas:
        - Create a depth threshold filter (near/far segmentation)
        - Calculate average distances to regions
        - Implement a "depth hole" filling algorithm
        - Create a "birds-eye view" projection
        """
        # TODO: Implement this function
        
        # HINT: Apply a distance threshold to find close objects
        # mask_near = cv2.inRange(depth_frame, 0, 1000)  # Objects within 1 meter
        
        # HINT: Calculate statistics about regions
        # mean_distance = np.mean(depth_frame[depth_frame > 0]) / 1000.0  # Convert to meters
        
        # HINT: Create a birds-eye view
        # height, width = depth_frame.shape
        # bird_view = np.zeros((height, width), dtype=np.uint8)
        # for y in range(height):
        #     for x in range(width):
        #         if depth_frame[y, x] > 0:
        #             # Map 3D position to 2D bird view
        #             bird_x = x
        #             bird_y = int(height - depth_frame[y, x]/50)  # Scale factor
        #             if 0 <= bird_y < height:
        #                 bird_view[bird_y, bird_x] = 255
        
        # For now, just publish the colormap
        if depth_colormap is not None:
            try:
                msg = self.bridge.cv2_to_imgmsg(depth_colormap, "bgr8")
                self.processed_depth_pub.publish(msg)
            except Exception as e:
                rospy.logerr("Error publishing processed depth image: {}".format(e))
    
    def process_rgbd_data(self, color_frame, depth_frame):
        """
        STUDENT TASK: Process combined RGB and depth data
        
        Learn about:
        - Aligning color and depth information
        - Object detection with RGBD data
        - Distance-based filtering of color objects
        
        Parameters:
        - color_frame: The RGB image
        - depth_frame: The depth image (same resolution as color_frame)
        
        Implementation ideas:
        - Segment objects by color, then get their distances
        - Create a colored point cloud visualization
        - Implement simple gesture recognition
        - Create a "green screen" effect using depth thresholds
        """
        # TODO: Implement this function
        
        # HINT: Color-based segmentation with depth filtering
        # hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        # lower_blue = np.array([100, 50, 50])
        # upper_blue = np.array([130, 255, 255])
        # blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # HINT: Get distances to blue objects
        # blue_depth = depth_frame.copy()
        # blue_depth[blue_mask == 0] = 0
        # if np.any(blue_depth > 0):
        #     avg_distance = np.mean(blue_depth[blue_depth > 0]) / 1000.0
        #     rospy.loginfo(f"Average distance to blue objects: {avg_distance:.2f} meters")
        
        # HINT: Create a depth-based green screen
        # background = np.ones_like(color_frame) * [0, 255, 0]  # Green background
        # mask = depth_frame > 1500  # Background starts at 1.5 meters
        # mask = np.stack([mask, mask, mask], axis=2)
        # mixed_image = np.where(mask, background, color_frame)
        
        # For now, just log that the function was called
        rospy.loginfo_throttle(5, "process_rgbd_data called - waiting for student implementation")
    
    def process_pointcloud(self, pointcloud_data):
        """
        STUDENT TASK: Process the point cloud data
        
        Learn about:
        - 3D point cloud processing
        - Plane detection
        - Object clustering in 3D
        - Coordinate transformations
        
        Parameters:
        - pointcloud_data: The raw point cloud data from the camera
        
        Implementation ideas:
        - Extract all points and store in a NumPy array
        - Find the floor plane using RANSAC
        - Cluster points to identify separate objects
        - Transform point coordinates to a different reference frame
        """
        # TODO: Implement this function
        
        # HINT: Convert point cloud to numpy array
        # points_list = []
        # for p in pc2.read_points(pointcloud_data, field_names=("x", "y", "z"), skip_nans=True):
        #     points_list.append(p)
        # points = np.array(points_list)
        
        # HINT: Find points within a certain region
        # region_mask = (points[:, 0] > -1) & (points[:, 0] < 1) & \
        #               (points[:, 1] > -1) & (points[:, 1] < 1) & \
        #               (points[:, 2] > 0) & (points[:, 2] < 2)
        # region_points = points[region_mask]
        
        # HINT: Calculate centroid of points
        # if len(region_points) > 0:
        #     centroid = np.mean(region_points, axis=0)
        #     rospy.loginfo(f"Centroid of region: x={centroid[0]:.2f}, y={centroid[1]:.2f}, z={centroid[2]:.2f}")
        
        # For now, just log that the function was called
        rospy.loginfo_throttle(5, "process_pointcloud called - waiting for student implementation")
    
    def detect_objects(self, color_frame, depth_frame):
        """
        STUDENT TASK: Detect objects using both color and depth information
        
        Learn about:
        - Object detection algorithms
        - Distance-based filtering
        - Connected component analysis
        
        Parameters:
        - color_frame: The RGB image
        - depth_frame: The depth image
        
        Implementation ideas:
        - Use contour detection on color-based segmentation
        - Filter contours based on size and depth
        - Implement a simple object tracker
        """
        # TODO: Implement this function
        
        # HINT: Basic contour detection
        # gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # HINT: Filter and process contours
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     if area > self.min_object_size:
        #         # Get bounding rectangle
        #         x, y, w, h = cv2.boundingRect(contour)
        #         
        #         # Get average depth in this region
        #         roi_depth = depth_frame[y:y+h, x:x+w]
        #         if np.any(roi_depth > 0):  # Avoid division by zero
        #             avg_depth = np.mean(roi_depth[roi_depth > 0]) / 1000.0  # Convert to meters
        #             
        #             # Draw rectangle and depth information
        #             cv2.rectangle(color_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #             cv2.putText(color_frame, f"{avg_depth:.2f}m", (x, y-10),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # For now, just log that the function was called
        rospy.loginfo_throttle(5, "detect_objects called - waiting for student implementation")
        
    def run(self):
        """Main loop"""
        rospy.loginfo("RealSense D435 handler running")
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        handler = RealSenseHandler()
        handler.run()
    except rospy.ROSInterruptException:
        pass