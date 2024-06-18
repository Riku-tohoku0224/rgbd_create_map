#!/usr/bin/env python3

import rospy
import ros_numpy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped,Point
import pandas as pd
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
from std_msgs.msg import Header, ColorRGBA
import cv2
from scipy.spatial.transform import Rotation as R
from bisect import bisect_left
from visualization_msgs.msg import Marker

tottori_map = o3d.geometry.PointCloud()
camera_positions = []
callback_counter = 0


def create_marker(points, frame_id):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "trajectory"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)

    # Apply transformation matrix for -15 degree rotation around the x-axis
    for point in points:
        p = np.array([point[0], point[1], point[2], 1])
        p_transformed =  p  # Apply the transformation matrix
        marker_point = Point()
        marker_point.x = p_transformed[0]
        marker_point.y = p_transformed[1]
        marker_point.z = p_transformed[2]
        marker.points.append(marker_point)
    
    return marker

def pose_to_matrix(pose_stamped):
    pose = pose_stamped.pose
    quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    translation = [pose.position.x, pose.position.y, pose.position.z]
    rotation = R.from_quat(quaternion).as_matrix()
    return rotation, translation

def pose_generator(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()  # カラム名の前後のスペースを削除
    poses = []
    for index, row in df.iterrows():
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = rospy.Time.from_sec(row['% UNIX time'])
        pose.pose.position.x = row['x(m)']
        pose.pose.position.y = row['y(m)']
        pose.pose.position.z = row['z(m)']
        pose.pose.orientation.w = row['orientation.w']
        pose.pose.orientation.x = row['orientation.x']
        pose.pose.orientation.y = row['orientation.y']
        pose.pose.orientation.z = row['orientation.z']
        poses.append(pose)
    return poses


def find_closest_pose(poses, target_time):
    times = [pose.header.stamp.to_sec() for pose in poses]
    idx = bisect_left(times, target_time.to_sec())
    if idx == 0:
        return poses[0]
    if idx == len(times):
        return poses[-1]
    before = times[idx - 1]
    after = times[idx]
    if target_time.to_sec() - before < after - target_time.to_sec():
        return poses[idx - 1]
    else:
        return poses[idx]

class PoseStampedSubscriber(message_filters.SimpleFilter):
    def __init__(self, csv_file):
        super(PoseStampedSubscriber, self).__init__()
        self.poses = pose_generator(csv_file)
        rospy.loginfo(f"PoseStampedSubscriber initialized with file: {csv_file}")

    def find_pose(self, target_time):
        return find_closest_pose(self.poses, target_time)

def images_callback(color_img, depth_img, pub, pose_subscriber, marker_pub ,frame_id):
    global callback_counter
    callback_counter += 1

    bridge = CvBridge()
    try:
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        # カラー画像の変換とログ出力
        cv_color_img = bridge.imgmsg_to_cv2(color_img, "bgr8")

        # カラー画像のリサイズを追加
        cv_color_img = cv2.resize(cv_color_img, (1032, 772), interpolation=cv2.INTER_LINEAR)

        # 深度画像の変換とログ出力
        cv_depth_img = bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        cv_depth_img = cv2.resize(cv_depth_img, (1032, 772), interpolation=cv2.INTER_NEAREST)
        cv_depth_img = np.array(cv_depth_img, dtype=np.float32) 

        # OpenCVのBGR形式をRGB形式に変換
        #cv_color_img = cv2.cvtColor(cv_color_img, cv2.COLOR_BGR2RGB)

        # OpenCVの画像をOpen3Dの画像形式に変換
        color_raw = o3d.geometry.Image(cv_color_img)
        depth_raw = o3d.geometry.Image(cv_depth_img)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False
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

        # 点群をNumpy配列に変換
        points = np.asarray(pcd.points)

        #IMUとカメラの座標系の違いを補正
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = 0.210
        pose.pose.position.y = 0
        pose.pose.position.z = 0.080
        pose.pose.orientation.x = 0.610
        pose.pose.orientation.y = -0.604
        pose.pose.orientation.z =  0.363
        pose.pose.orientation.w = -0.362

        rotation, translation = pose_to_matrix(pose)
        points = (np.dot(rotation, points.T).T + translation)
        
        # 最も近いポーズデータを取得
        closest_pose = pose_subscriber.find_pose(color_img.header.stamp)
        rospy.loginfo(f"Closest PoseStamped: {closest_pose}")
        
        # カメラの姿勢の行列を取得
        T_rotation, T_translation = pose_to_matrix(closest_pose)
        print(closest_pose)
        points = (np.dot(T_rotation, points.T).T + T_translation)
       



        # 変換された点群をOpen3Dの点群オブジェクトに変換
        pcd.points = o3d.utility.Vector3dVector(points)

        global tottori_map
        global camera_positions

        # tottori_mapに点群を追加
        tottori_map += pcd
        voxel_size = 0.01
        tottori_map = tottori_map.voxel_down_sample(voxel_size=voxel_size)

               # カメラ位置を追加
        camera_positions.append([closest_pose.pose.position.x,
                                 closest_pose.pose.position.y - 0.5,
                                 closest_pose.pose.position.z])
        marker = create_marker(camera_positions, frame_id)
        
        # ログ出力
        rospy.loginfo(f"ボクセルダウンサンプリング後の点群には {len(tottori_map.points)} 点")

        # 10回ごとにPointCloud2メッセージをpublish
        if callback_counter % 10 == 0:
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
            marker_pub.publish(marker)
            rospy.loginfo(f"PointCloud2メッセージを {len(points)} 点で公開")

    except Exception as e:
        rospy.logerr(f"処理に失敗: {e}")


def listener():
    rospy.init_node('create_dense_map', anonymous=False)
    pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
    marker_pub = rospy.Publisher('/camera_trajectory', Marker, queue_size=10)
    frame_id = "camera_left"

    color_sub = message_filters.Subscriber("/hcru2/pt_color_rect/left/image", Image)
    depth_sub = message_filters.Subscriber("/hcru2/pt_stereo_sgm/depth", Image)
    pose_sub = PoseStampedSubscriber('/home/riku-suzuki/madmax/B-6_ground_truth/gt_6DoF_gnss_and_imu.csv')

    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
    ts.registerCallback(lambda color, depth: images_callback(color, depth, pub, pose_sub, marker_pub ,frame_id))

    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception occurred")
