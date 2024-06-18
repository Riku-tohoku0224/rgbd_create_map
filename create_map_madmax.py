#!/usr/bin/env python3

import rospy
import ros_numpy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
import pandas as pd
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
from std_msgs.msg import Header
import cv2
from scipy.spatial.transform import Rotation as R

tottori_map = o3d.geometry.PointCloud()
camera_positions = []
callback_counter = 0

def pose_to_matrix(pose_stamped):
    pose = pose_stamped.pose
    quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    translation = [pose.position.x, pose.position.y, pose.position.z]
    rotation = R.from_quat(quaternion).as_matrix()
    return rotation, translation

def images_callback(color_img, depth_img, pose_msg, pub, frame_id):
    global callback_counter
    callback_counter += 1

    rospy.loginfo("同期した画像とカメラのポーズを受信")
    bridge = CvBridge()
    try:
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        # カラー画像の変換とログ出力
        cv_color_img = bridge.imgmsg_to_cv2(color_img, "bgr8")
        rospy.loginfo(f"カラー画像の形式: {cv_color_img.dtype}, 形状: {cv_color_img.shape}")
        
        # カラー画像のリサイズを追加
        cv_color_img = cv2.resize(cv_color_img, (1032, 772), interpolation=cv2.INTER_LINEAR)
        rospy.loginfo(f"リサイズ後のカラー画像の形式: {cv_color_img.dtype}, 形状: {cv_color_img.shape}")

        # 深度画像の変換とログ出力
        cv_depth_img = bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        cv_depth_img = cv2.resize(cv_depth_img, (1032, 772), interpolation=cv2.INTER_NEAREST)
        cv_depth_img = np.array(cv_depth_img, dtype=np.float32) * 1000.0
        rospy.loginfo(f"変換前の深度画像の形式: {cv_depth_img.dtype}, 形状: {cv_depth_img.shape}")

        # OpenCVのBGR形式をRGB形式に変換
        cv_color_img = cv2.cvtColor(cv_color_img, cv2.COLOR_BGR2RGB)
        rospy.loginfo(f"RGB形式に変換後のカラー画像の形式: {cv_color_img.dtype}, 形状: {cv_color_img.shape}")

        # OpenCVの画像をOpen3Dの画像形式に変換
        color_raw = o3d.geometry.Image(cv_color_img)
        depth_raw = o3d.geometry.Image(cv_depth_img)
        rospy.loginfo(f"Open3Dカラー画像の形式: {np.asarray(color_raw).dtype}, 形状: {np.asarray(color_raw).shape}")
        rospy.loginfo(f"Open3D深度画像の形式: {np.asarray(depth_raw).dtype}, 形状: {np.asarray(depth_raw).shape}")

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=1032,
            height=772,
            fx=864.387610,
            fy=863.858468,
            cx=519.731844,
            cy=385.718642
        )

        # RGBD画像から点群を生成
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.transform([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

        global tottori_map
        global camera_positions

        # tottori_mapに点群を追加
        tottori_map += pcd
        voxel_size = 0.00001
        tottori_map = tottori_map.voxel_down_sample(voxel_size=voxel_size)
        
        # ログ出力
        rospy.loginfo(f"ボクセルダウンサンプリング後の点群には {len(tottori_map.points)} 点")

        # 100回ごとにPointCloud2メッセージをpublish
        if callback_counter % 1 == 0:
            points = np.asarray(tottori_map.points)
            colors = np.asarray(tottori_map.colors) * 255
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
        rospy.logerr(f"処理に失敗: {e}")

def pose_generator(csv_file):
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = rospy.Time.from_sec(row['% UNIX time'])
        pose.pose.position.x = row[' x(m)']
        pose.pose.position.y = row[' y(m)']
        pose.pose.position.z = row[' z(m)']
        pose.pose.orientation.w = row[' orientation.w']
        pose.pose.orientation.x = row[' orientation.x']
        pose.pose.orientation.y = row[' orientation.y']
        pose.pose.orientation.z = row[' orientation.z']
        yield pose

class PoseStampedSubscriber(message_filters.SimpleFilter):
    def __init__(self, csv_file):
        super(PoseStampedSubscriber, self).__init__()
        self.pose_gen = pose_generator(csv_file)

    def get(self):
        try:
            pose = next(self.pose_gen)
            self.signalMessage(pose)
        except StopIteration:
            rospy.signal_shutdown('No more poses in CSV file')

def listener():
    rospy.init_node('create_dense_map', anonymous=False)
    pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
    frame_id = "camera_left"

    color_sub = message_filters.Subscriber("/hcru2/pt_color_rect/left/image", Image)
    depth_sub = message_filters.Subscriber("/hcru2/pt_stereo_sgm/depth", Image)
    pose_sub = PoseStampedSubscriber('/home/riku-suzuki/madmax/B-6_ground_truth/gt_6DoF_gnss_and_imu.csv')

    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, pose_sub], 10, 0.1)
    ts.registerCallback(lambda color, depth, pose: images_callback(color, depth, pose, pub, frame_id))

    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
