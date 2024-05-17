#!/usr/bin/env python3

import rospy
import ros_numpy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
from std_msgs.msg import Header
import cv2

tottori_map = o3d.geometry.PointCloud()

def pose_to_matrix(pose_stamped):
    pose = pose_stamped.pose
    # Quaternion order might need to be verified: (w, x, y, z)
    quaternion = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    translation = [pose.position.x, pose.position.y, pose.position.z]
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def images_callback(color_img, depth_img, camera_pose_stamped, pub, frame_id):
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
            color_raw, depth_raw, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 616.5249633789062, 616.7235717773438, 331.1578674316406, 234.31881713867188
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.transform([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        T = pose_to_matrix(camera_pose_stamped)
        pcd.transform(T)

        global tottori_map
        tottori_map += pcd

        voxel_size = 000000.1
        downsampled_tottori_map = tottori_map.voxel_down_sample(voxel_size)
        rospy.loginfo(f"ボクセルダウンサンプリング後の点群には {len(downsampled_tottori_map.points)} 点があります")

        points = np.asarray(downsampled_tottori_map.points)
        colors = np.asarray(downsampled_tottori_map.colors) * 255
        colors = colors.astype(np.uint8)

        # エンディアンを考慮した色のエンコード
        rgb_colors = np.array([((r << 16) | (g << 8) | b) for b, g, r in colors])

        pc_array = np.zeros(len(points), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)])
        pc_array['x'] = points[:, 0]
        pc_array['y'] = points[:, 1]
        pc_array['z'] = points[:, 2]
        pc_array['rgb'] = rgb_colors

        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=header.stamp, frame_id=header.frame_id)
        pub.publish(pc_msg)
        rospy.loginfo(f"PointCloud2メッセージを {len(points)} 点で公開しました")

    except Exception as e:
        rospy.logerr(f"画像とポーズの処理に失敗しました: {e}")



def listener():
    rospy.init_node('create_dense_map', anonymous=False)
    pub = rospy.Publisher('/tottori_map_point_cloud', PointCloud2, queue_size=10)
    frame_id = "map"

    color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
    pose_sub = message_filters.Subscriber("/orb_slam3/camera_pose", PoseStamped)

    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, pose_sub], 10, 0.1)
    ts.registerCallback(lambda color, depth, pose_stamped: images_callback(color, depth, pose_stamped, pub, frame_id))

    rospy.spin()

if __name__ == '__main__':
    listener()