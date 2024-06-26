#!/usr/bin/env python3

import rospy
import ros_numpy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, Point
import pandas as pd
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
from std_msgs.msg import Header, ColorRGBA
import cv2
from scipy.spatial.transform import Rotation as R
from bisect import bisect_left
from visualization_msgs.msg import Marker
import threading
import os
from datetime import datetime

map = o3d.geometry.PointCloud()
previous_map = o3d.geometry.PointCloud()  # 前の点群データを保持するための変数
camera_positions = []
callback_counter = 0
save_flag = False
PointCloud_save_directory = "/home/riku-suzuki/madmax/test/C-0/10"

data_size_directory = "/home/riku-suzuki/madmax/test/C-0/10"
# 現在の日時を取得
now = datetime.now()
# 日時を文字列に変換
timestamp_str = now.strftime("%Y%m%d_%H%M%S")

# ファイル名に日時を含める
data_size_file = os.path.join(data_size_directory, f"data_size_{timestamp_str}.csv")

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

    for point in points:
        marker_point = Point()
        marker_point.x, marker_point.y, marker_point.z = point
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
    df.columns = df.columns.str.strip()
    poses = []
    for _, row in df.iterrows():
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

def apply_low_pass_filter_to_point_cloud(pcd, base_height, resolution, filter_size):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]

    z_data = z_data - base_height

    x_edges = np.arange(x_data.min(), x_data.max() + resolution, resolution)
    y_edges = np.arange(y_data.min(), y_data.max() + resolution, resolution)

    z_grid = np.full((len(x_edges) - 1, len(y_edges) - 1, 4), np.nan, dtype=np.float32)
    valid_grid = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=bool)
    
    for i in range(len(x_data)):
        x_idx = np.searchsorted(x_edges, x_data[i]) - 1
        y_idx = np.searchsorted(y_edges, y_data[i]) - 1
        if np.isnan(z_grid[x_idx, y_idx, 0]) or abs(z_data[i]) > abs(z_grid[x_idx, y_idx, 0]):
            z_grid[x_idx, y_idx, 0] = z_data[i]
            z_grid[x_idx, y_idx, 1] = colors[i, 0]
            z_grid[x_idx, y_idx, 2] = colors[i, 1]
            z_grid[x_idx, y_idx, 3] = colors[i, 2]
            valid_grid[x_idx, y_idx] = True

    z_grid[np.isnan(z_grid[:, :, 0])] = 0
    valid_grid[np.isnan(z_grid[:, :, 0])] = False

    rows, cols, _ = z_grid.shape

    valid_grid_memory_usage = valid_grid.size  # bool は 1 バイト
    f = np.fft.fft2(np.nan_to_num(z_grid[:,:,0]))
    fshift = np.fft.fftshift(f)
   
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i - crow)**2 + (j - ccol)**2 <= filter_size**2:
                mask[i, j] = 1

    fshift_masked = fshift * mask

    non_zero_before = np.count_nonzero(fshift)
    non_zero_after = np.count_nonzero(fshift_masked)
    fshift_total_memory_usage = non_zero_before * 8
    fshift_masked_total_memory_usage = non_zero_after * 8

    print(f"Total memory usage of valid_grid: {valid_grid_memory_usage} bytes")
    print(f"Total memory usage of fshift: {fshift_total_memory_usage} bytes")
    print(f"Total memory usage of fshift_masked: {fshift_masked_total_memory_usage} bytes")
    print(f"Data size before filtering: {fshift_total_memory_usage + valid_grid_memory_usage} bytes")
    print(f"Data size after filtering : {fshift_masked_total_memory_usage + valid_grid_memory_usage} bytes")
    # ファイルが存在するか確認
    file_exists = os.path.exists(data_size_file)
    
    # データサイズの記録
    data_size_record = {
        "callback_counter": callback_counter,
        "valid_grid_memory_usage": valid_grid_memory_usage,
        "fshift_total_memory_usage": fshift_total_memory_usage,
        "fshift_masked_total_memory_usage": fshift_masked_total_memory_usage,
        "total_memory_usage_before_filtering": fshift_total_memory_usage + valid_grid_memory_usage,
        "total_memory_usage_after_filtering": fshift_masked_total_memory_usage + valid_grid_memory_usage
    }

    # CSVファイルに追記
    df = pd.DataFrame([data_size_record])

    # ファイルが存在しない場合はヘッダーを付けて書き込む
    if not file_exists:
        df.to_csv(data_size_file, mode='a', header=True, index=False)
    else:
        df.to_csv(data_size_file, mode='a', header=False, index=False)

    print(f"Data size record appended to {data_size_file}")

    return fshift_masked, x_edges, y_edges, z_grid, valid_grid

def regenerate_point_cloud(fshift_masked, x_edges, y_edges, z_grid, valid_grid, base_height):
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    filtered_z_grid = np.zeros_like(z_grid)
    filtered_z_grid[:,:,0] = img_back
    filtered_z_grid[:,:,1:] = z_grid[:,:,1:]

    mask = valid_grid.ravel()
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing='ij')

    new_points = np.vstack((x_coords.ravel()[mask], y_coords.ravel()[mask], filtered_z_grid[:,:,0].ravel()[mask] + base_height)).T
    new_colors = np.vstack((filtered_z_grid[:,:,1].ravel()[mask], filtered_z_grid[:,:,2].ravel()[mask], filtered_z_grid[:,:,3].ravel()[mask])).T

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(new_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    return filtered_pcd

def key_listener():
    global save_flag
    while not rospy.is_shutdown():
        key = input()
        if key == 's':
            save_flag = True

def images_callback(color_img, depth_img, pub, pose_subscriber, marker_pub, frame_id):
    global callback_counter, save_flag, map, previous_map
    callback_counter += 1

    bridge = CvBridge()
    try:
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        cv_color_img = bridge.imgmsg_to_cv2(color_img, "bgr8")
        cv_color_img = cv2.resize(cv_color_img, (1032, 772), interpolation=cv2.INTER_LINEAR)

        cv_depth_img = bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        cv_depth_img = cv2.resize(cv_depth_img, (1032, 772), interpolation=cv2.INTER_NEAREST)
        cv_depth_img = np.array(cv_depth_img, dtype=np.float32)

        color_raw = o3d.geometry.Image(cv_color_img)
        depth_raw = o3d.geometry.Image(cv_depth_img)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=1032,
            height=772,
            fx=868.246,
            fy=868.246,
            cx=516.0,
            cy=386.0
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        points = np.asarray(pcd.points)
        
        tf_B_to_camera = np.array([ [ 0.00656078, -0.47419651,  0.88039459, 0.210],
                                    [-0.99996299, -0.00801117,  0.00313685, 0.0],
                                    [ 0.00556551, -0.88038258, -0.47423152, 0.080],
                                    [ 0.0       ,  0.0       ,  0.0       , 1.0]])

        rotation = tf_B_to_camera[:3, :3]
        translation = tf_B_to_camera[:3, 3]
        points = (np.dot(rotation, points.T).T + translation)
        
        closest_pose = pose_subscriber.find_pose(color_img.header.stamp)
        
        T_rotation, T_translation = pose_to_matrix(closest_pose)
        points = (np.dot(T_rotation, points.T).T + T_translation)
        
        pcd.points = o3d.utility.Vector3dVector(points)

        centroid = np.asarray(pcd.points).mean(axis=0)
        height_every_frame = centroid[2]

        resolution = 0.1
        filter_size = 10
        fshift_masked, x_edges, y_edges, z_grid, valid_grid = apply_low_pass_filter_to_point_cloud(pcd, height_every_frame, resolution, filter_size)
        pcd = regenerate_point_cloud(fshift_masked, x_edges, y_edges, z_grid, valid_grid, height_every_frame)

        previous_map = map  # 現在のmapをprevious_mapに保存
        map += pcd
        voxel_size = 0.1
        map = map.voxel_down_sample(voxel_size=voxel_size)

        camera_positions.append([closest_pose.pose.position.x,
                                 closest_pose.pose.position.y,
                                 closest_pose.pose.position.z])
        marker = create_marker(camera_positions, frame_id)

        rospy.loginfo(f"ボクセルダウンサンプリング後の点群には {len(map.points)} 点")

        if callback_counter % 50 == 0:
            points = np.asarray(map.points)
            colors = np.asarray(map.colors) * 255
            colors = colors.astype(np.uint8)

            assert len(points) == len(colors), "pointsとcolorsの長さが一致しません"

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

        # 画像と点群の保存処理
        if save_flag:
            if not os.path.exists(PointCloud_save_directory):
                os.makedirs(PointCloud_save_directory)
            pcd_path = os.path.join(PointCloud_save_directory, f"map_{callback_counter - 1}.pcd")
            o3d.io.write_point_cloud(pcd_path, previous_map)
            rospy.loginfo(f"{pcd_path} を保存しました")
            save_flag = False

    except Exception as e:
        rospy.logerr(f"処理に失敗: {e}")

def listener():
    rospy.init_node('create_dense_map', anonymous=False)
    pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
    marker_pub = rospy.Publisher('/camera_trajectory', Marker, queue_size=10)
    frame_id = "camera_left"

    color_sub = message_filters.Subscriber("/hcru2/pt_color_rect/left/image", Image)
    depth_sub = message_filters.Subscriber("/hcru2/pt_stereo_sgm/depth", Image)
    pose_sub = PoseStampedSubscriber('/home/riku-suzuki/madmax/C-0_ground_truth/cleaned2_data.csv')

    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
    ts.registerCallback(lambda color, depth: images_callback(color, depth, pub, pose_sub, marker_pub, frame_id))

    key_thread = threading.Thread(target=key_listener)
    key_thread.daemon = True
    key_thread.start()

    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception occurred")
