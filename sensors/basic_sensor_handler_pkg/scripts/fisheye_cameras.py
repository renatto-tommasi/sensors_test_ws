#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
from sensor_msgs.msg import CameraInfo


class FisheyeCameraNode:
    def __init__(self):
        rospy.init_node('fisheye_camera_node', anonymous=True)
        
        # Camera parameters
        self.camera_names = ['front', 'rear', 'right', 'left']
        self.num_cameras = len(self.camera_names)
        self.camera_topics = [f'/fisheye_{name}/image_raw' for name in self.camera_names]
        self.camera_images = [None] * self.num_cameras
        
        # Camera info topics
        self.camera_info_topics = [f'/fisheye_{name}/camera_info' for name in self.camera_names]
        self.camera_info = [None] * self.num_cameras
        
        # Initialize CV bridge for converting between ROS Image messages and OpenCV images
        self.bridge = CvBridge()
        
        # Initialize subscribers for image and camera info
        self.image_subscribers = []
        self.info_subscribers = []
        
        for i, (image_topic, info_topic) in enumerate(zip(self.camera_topics, self.camera_info_topics)):
            self.image_subscribers.append(
                rospy.Subscriber(image_topic, Image, self.image_callback, callback_args=i)
            )
            self.info_subscribers.append(
                rospy.Subscriber(info_topic, CameraInfo, self.camera_info_callback, callback_args=i)
            )
        
        # Output directory for processed images
        self.output_dir = os.path.join(os.path.expanduser("~"), "fisheye_processed")
        os.makedirs(self.output_dir, exist_ok=True)
        
        rospy.loginfo("Fisheye camera node initialized")
    
    def image_callback(self, msg, camera_index):
        """Callback function for image messages"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.camera_images[camera_index] = cv_image
            rospy.loginfo(f"Received image from camera {self.camera_names[camera_index]}")
            
            # Process the image - students can implement these methods
            self.process_fisheye_image(camera_index)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
    
    def camera_info_callback(self, msg, camera_index):
        """Callback function for camera info messages"""
        self.camera_info[camera_index] = msg
        rospy.loginfo(f"Received camera info for {self.camera_names[camera_index]}")
    
    # Placeholder methods for students to implement
    
    def process_fisheye_image(self, camera_index):
        """Process the fisheye image from the specified camera
        
        This is a placeholder method for students to implement
        """
        if self.camera_images[camera_index] is not None:
            # TODO: Students to implement fisheye image processing
            # Some ideas:
            # - Undistort the fisheye image
            # - Apply filters or transformations
            # - Display the raw and processed images
            
            # Here's a simple example that just displays the raw image
            self.display_image(self.camera_images[camera_index], self.camera_names[camera_index])
            
            # Try to create a panorama if we have all images
            if all(img is not None for img in self.camera_images):
                self.create_panorama()
    
    def undistort_fisheye_image(self, camera_index):
        """Undistort the fisheye image using camera calibration parameters
        
        This is a placeholder method for students to implement
        """
        if self.camera_images[camera_index] is None or self.camera_info[camera_index] is None:
            return None
        
        # TODO: Students to implement fisheye undistortion
        # Steps:
        # 1. Extract camera matrix (K) and distortion coefficients (D) from camera_info
        # 2. Use cv2.fisheye.undistortImage to undistort the image
        
        # For now, just return the original image
        return self.camera_images[camera_index]
    
    def create_panorama(self):
        """Create a panorama from all four fisheye cameras
        
        This is a placeholder method for students to implement
        """
        # TODO: Students to implement panorama creation
        # Steps:
        # 1. Undistort all fisheye images
        # 2. Stitch the images together using cv2.Stitcher or manually
        
        # For now, just create a simple grid of the four images
        if all(img is not None for img in self.camera_images):
            # Resize images to a common size
            size = (300, 300)
            resized_images = [cv2.resize(img, size) for img in self.camera_images]
            
            # Create a 2x2 grid
            top_row = np.hstack((resized_images[0], resized_images[1]))
            bottom_row = np.hstack((resized_images[2], resized_images[3]))
            panorama = np.vstack((top_row, bottom_row))
            
            self.display_image(panorama, "panorama")
    
    def display_image(self, image, name):
        """Display an image (for debugging purposes)"""
        # In ROS, it's better to publish processed images rather than use cv2.imshow
        # But for debugging, we can save the images
        cv2.imwrite(os.path.join(self.output_dir, f"{name}.jpg"), image)

def main():
    """Main function to initialize and run the node"""
    try:
        node = FisheyeCameraNode()
        rospy.loginfo("Fisheye camera node running. Press Ctrl+C to terminate.")
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down fisheye camera node")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()