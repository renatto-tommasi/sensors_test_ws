#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf2_ros
import tf2_geometry_msgs


import sensor_msgs.point_cloud2 as pc2

class LidarProcessor:
    def __init__(self):
        rospy.init_node('lidar_processor', anonymous=True)
        
        # Parameters
        self.point_cloud_topic = rospy.get_param('~point_cloud_topic', '/velodyne_points')
        
        # Publishers
        self.marker_pub = rospy.Publisher('/lidar_markers', MarkerArray, queue_size=10)
        self.filtered_cloud_pub = rospy.Publisher('/filtered_points', PointCloud2, queue_size=10)
        
        # Subscribers
        rospy.Subscriber(self.point_cloud_topic, PointCloud2, self.pointcloud_callback)
        
        rospy.loginfo("LiDAR processor node initialized")
    
    def pointcloud_callback(self, msg):
        """
        Process incoming pointcloud messages
        """
        # Convert PointCloud2 to numpy array
        points = self.pointcloud2_to_xyz(msg)
        
        # TODO: Students implement data processing pipeline
        filtered_points = self.filter_pointcloud(points)
        downsampled_points = self.downsample_pointcloud(filtered_points)
        clusters = self.cluster_pointcloud(downsampled_points)
        ground_removed = self.remove_ground_plane(downsampled_points)
        
        # Visualize results
        self.visualize_clusters(clusters, msg.header)
        
        rospy.loginfo(f"Processed cloud with {len(points)} points -> {len(filtered_points)} filtered -> {len(clusters)} clusters")
    
    def pointcloud2_to_xyz(self, cloud_msg):
        """
        Convert a PointCloud2 message to a numpy array of XYZ coordinates
        """
        # Extract points from PointCloud2 message
        pc_data = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        return np.array(list(pc_data))
    
    def filter_pointcloud(self, points):
        """
        TODO: Students implement this function
        
        Filter the pointcloud to remove noise and outliers
        
        Args:
            points: Nx3 numpy array of XYZ points
            
        Returns:
            Mx3 numpy array of filtered XYZ points
        
        Implementation suggestions:
        - Remove points beyond a certain distance
        - Remove statistical outliers
        - Apply a voxel grid filter
        """
        # Example implementation (students should replace this)
        # Simple distance-based filter
        distances = np.sqrt(np.sum(points**2, axis=1))
        max_distance = 50.0  # meters
        return points[distances < max_distance]
    
    def downsample_pointcloud(self, points):
        """
        TODO: Students implement this function
        
        Downsample the pointcloud to reduce processing time
        
        Args:
            points: Nx3 numpy array of XYZ points
            
        Returns:
            Mx3 numpy array of downsampled XYZ points where M < N
        
        Implementation suggestions:
        - Random downsampling
        - Voxel grid downsampling
        - Uniform sampling
        """
        # Example implementation (students should replace this)
        # Simple random downsampling
        if len(points) > 1000:
            indices = np.random.choice(len(points), 1000, replace=False)
            return points[indices]
        return points
    
    def cluster_pointcloud(self, points):
        """
        TODO: Students implement this function
        
        Cluster the pointcloud to identify distinct objects
        
        Args:
            points: Nx3 numpy array of XYZ points
            
        Returns:
            List of numpy arrays, each representing a cluster
        
        Implementation suggestions:
        - DBSCAN clustering
        - Euclidean clustering
        - Region growing
        """
        # Example implementation (students should replace this)
        # Simple distance-based clustering (not effective, just for demonstration)
        clusters = []
        remaining_points = points.copy()
        
        while len(remaining_points) > 10:  # Arbitrary minimum cluster size
            # Start a new cluster with the first point
            current_cluster = [remaining_points[0]]
            remaining_points = remaining_points[1:]
            
            # Find points close to any point in the current cluster
            i = 0
            while i < len(remaining_points):
                # Check if this point is close to any point in the current cluster
                min_dist = min(np.linalg.norm(remaining_points[i] - p) for p in current_cluster[:10])  # Check only first 10 points for efficiency
                if min_dist < 0.5:  # 0.5m threshold
                    current_cluster.append(remaining_points[i])
                    remaining_points = np.delete(remaining_points, i, axis=0)
                else:
                    i += 1
                    
            if len(current_cluster) > 10:
                clusters.append(np.array(current_cluster))
                
            # Safety check to prevent infinite loops
            if len(clusters) > 20:
                break
                
        return clusters
    
    def remove_ground_plane(self, points):
        """
        TODO: Students implement this function
        
        Remove ground plane points from the pointcloud
        
        Args:
            points: Nx3 numpy array of XYZ points
            
        Returns:
            Mx3 numpy array with ground points removed
        
        Implementation suggestions:
        - RANSAC plane fitting
        - Height-based thresholding
        - Normal-based segmentation
        """
        # Example implementation (students should replace this)
        # Simple height-based filtering
        ground_threshold = -1.5  # Assume ground is at z = -1.5 and below
        return points[points[:, 2] > ground_threshold]
    
    def visualize_clusters(self, clusters, header):
        """
        Visualize the clusters as marker arrays in RViz
        """
        marker_array = MarkerArray()
        
        for i, cluster in enumerate(clusters):
            marker = Marker()
            marker.header = header
            marker.ns = "clusters"
            marker.id = i
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            # Assign a different color to each cluster
            marker.color.r = np.random.random()
            marker.color.g = np.random.random()
            marker.color.b = np.random.random()
            marker.color.a = 1.0
            
            # Add points to the marker
            for point in cluster:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                marker.points.append(p)
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

def main():
    processor = LidarProcessor()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down LiDAR processor node")

if __name__ == '__main__':
    main()