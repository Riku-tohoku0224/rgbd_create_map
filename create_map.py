#!/usr/bin/env python3

import rospy
import ros_numpy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, Point  
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
from std_msgs.msg import Header, ColorRGBA
import cv2
from visualization_msgs.msg import Marker

tottori_map = o3d.geometry.PointCloud()
camera_positions = []

def pose_to_matrix(pose_stamped):
    pose = pose_stamped.pose
    quaternion = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    translation = [pose.position.x, pose.position.y, pose.position.z]
    
    # 回転行列を取得
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    
    # 4x4の単位行列を作成
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    
    return T

def apply_transform(point, transform_matrix):
    point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
    transformed_point = transform_matrix @ point_homogeneous
    return transformed_point[:3]

def create_marker(points, frame_id):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "trajectory"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # ラインの幅

    marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # 赤色

    for point in points:
        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = point[2]
        marker.points.append(p)
    
    return marker

def images_callback(color_img, depth_img, camera_pose_stamped, pub, marker_pub, frame_id):
    rospy.loginfo("同期した画像とカメラのポーズを受信")
    bridge = CvBridge()
    try:
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        cv_color_img = bridge.imgmsg_to_cv2(color_img, "bgr8")
        cv_depth_img = bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        cv_depth_img = cv2.resize(cv_depth_img, (640, 480), interpolation=cv2.INTER_NEAREST)
        cv_depth_img = np.array(cv_depth_img, dtype=np.float32) / 1000.0

        color_raw = o3d.geometry.Image(cv_color_img)
        depth_raw = o3d.geometry.Image(cv_depth_img)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 616.5249633789062, 616.7235717773438, 331.1578674316406, 234.31881713867188
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        T = pose_to_matrix(camera_pose_stamped)
        pcd.transform(T)

        # OpenCVの左手系からROSの右手系への変換
        pcd.transform([[1, 0, 0, 0], 
                       [0, -1, 0, 0], 
                       [0, 0, -1, 0], 
                       [0, 0, 0, 1]])

        global tottori_map
        global camera_positions
        tottori_map += pcd

        camera_positions.append([camera_pose_stamped.pose.position.x,
                                 camera_pose_stamped.pose.position.y,
                                 camera_pose_stamped.pose.position.z])
        marker = create_marker(camera_positions, frame_id)
        marker_pub.publish(marker)

        voxel_size = 0.1
        downsampled_tottori_map = tottori_map.voxel_down_sample(voxel_size)
        rospy.loginfo(f"ボクセルダウンサンプリング後の点群には {len(downsampled_tottori_map.points)} 点")

        points = np.asarray(downsampled_tottori_map.points)
        colors = np.asarray(downsampled_tottori_map.colors) * 255
        colors = colors.astype(np.uint8)

        rgb_colors = np.array([((r << 16) | (g << 8) | b) for b, g, r in colors])

        pc_array = np.zeros(len(points), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)])
        pc_array['x'] = points[:, 0]
        pc_array['y'] = points[:, 1]
        pc_array['z'] = points[:, 2]
        pc_array['rgb'] = rgb_colors

        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=header.stamp, frame_id=header.frame_id)
        pub.publish(pc_msg)
        rospy.loginfo(f"PointCloud2メッセージを {len(points)} 点で公開")

    except Exception as e:
        rospy.logerr(f"処理に失敗{e}")

def listener():
    rospy.init_node('create_dense_map', anonymous=False)
    pub = rospy.Publisher('/tottori_map_point_cloud', PointCloud2, queue_size=10)
    marker_pub = rospy.Publisher('/camera_trajectory', Marker, queue_size=10)
    frame_id = "map"

    color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
    pose_sub = message_filters.Subscriber("/orb_slam3/camera_pose", PoseStamped)

    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, pose_sub], 10, 0.1)
    ts.registerCallback(lambda color, depth, pose_stamped: images_callback(color, depth, pose_stamped, pub, marker_pub, frame_id))

    rospy.spin()

if __name__ == '__main__':
    listener()
