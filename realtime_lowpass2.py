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
from datetime import datetime

map = o3d.geometry.PointCloud()
camera_positions = []
callback_counter = 0
frequency_domain_data_list = []
#PointCloud_save_directory = "/home/riku-suzuki/madmax/test/E-2/original_each_frame_with_image"

#data_size_directory = "/home/riku-suzuki/madmax/test/E-2/original_each_frame_with_image"
# 現在の日時を取得
now = datetime.now()
# 日時を文字列に変換
timestamp_str = now.strftime("%Y%m%d_%H%M%S")

# ファイル名に日時を含める
#data_size_file = os.path.join(data_size_directory, f"data_size_{timestamp_str}.csv")

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

def create_arrow_marker(pose, frame_id):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "camera_orientation"
    marker.id = 1
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 0.5  # 矢印の長さ
    marker.scale.y = 0.05  # 矢印の幅
    marker.scale.z = 0.05  # 矢印の高さ
    marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # 矢印の色（緑色）

    marker.pose = pose.pose
    return marker


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
    
def calculate_edges(x_min, x_max, y_min, y_max, resolution):
    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)
    return x_edges, y_edges


def apply_low_pass_filter_to_point_cloud(pcd, base_height, resolution, filter_size):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]

     # x_min, x_max, y_min, y_maxを計算
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()


    x_edges, y_edges = calculate_edges(x_min, x_max, y_min, y_max, resolution)

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

    z_grid[np.isnan(z_grid[:, :, 0])] = base_height
    valid_grid[np.isnan(z_grid[:, :, 0])] = False

    rows, cols, _ = z_grid.shape
    # フィルタを適用する
    f = np.fft.fft2(np.nan_to_num(z_grid[:, :, 0]))
    z_grid[:, :, 0] = np.nan    # Zの空間領域での高さを忘れて、メモリを開放
    fshift = np.fft.fftshift(f)

    crow, ccol = int(rows / 2), int(cols / 2)
    
    # フィルタリングが適用される最大の filter_size を計算
    max_distance = np.sqrt(crow**2 + ccol**2)
    max_filter_size = max_distance * resolution * 10  # 周波数空間での距離を空間分解能に換算
    real_filter_size = (1 - filter_size) * max_filter_size
    print("real_filter_size", real_filter_size)

    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i - crow)**2 + (j - ccol)**2 <= real_filter_size**2:
                mask[i, j] = 1

    fshift_masked = fshift * mask
    print(f"fshift_masked:{fshift_masked.shape}")
    print(f"valid_grid:{valid_grid.shape}")

    # 円の内側のフィルタリングが欠けられていない部分のみを格納
    fshift_masked_data_reduction = fshift_masked[mask == 1]
    print(f"fshift_masked_data_reduction:{fshift_masked_data_reduction.shape}")

    x_center = x_edges[crow]
    y_center = y_edges[ccol]

    return fshift_masked_data_reduction, z_grid, valid_grid, x_min, x_max, y_min, y_max





def extract_position_from_pose(pose_stamped):
    # PoseStampedから位置情報を抽出
    position = pose_stamped.pose.position
    return np.array([position.x, position.y])

def find_data_within_radius(closest_pose, peripheral_radius):
    closest_position = extract_position_from_pose(closest_pose)
    found_data = []
    for data in frequency_domain_data_list:
        fshift_masked_data_reduction_real,fshift_masked_data_reduction_imag,z_grid, valid_grid,x_min,x_max,y_min,y_max = data
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        center = np.array([x_center, y_center])
        distance = np.linalg.norm(closest_position - center)
        
        if distance <= peripheral_radius:
            found_data.append(data)
    
    return found_data

def regenerate_point_cloud(closest_pose, peripheral_radius, filter_size, resolution):
    matching_data = find_data_within_radius(closest_pose, peripheral_radius)
    if not matching_data:
        rospy.logwarn("regenerate_point_cloud: No matching data found")
        return None  # 条件を満たすデータがない場合はNoneを返す

    new_points = []
    new_colors = []

    for data in matching_data:
        if len(data) != 8:
            rospy.logerr(f"regenerate_point_cloud: Data length mismatch, expected 8, got {len(data)}")
            continue
        fshift_masked_real, fshift_masked_imag, z_grid, valid_grid, x_min, x_max, y_min, y_max = data
        
        x_edges, y_edges = calculate_edges(x_min, x_max, y_min, y_max, resolution)

        fshift_masked = fshift_masked_real + 1j * fshift_masked_imag
        rows, cols = valid_grid.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        # 復元するためのマスクを再生成
        max_distance = np.sqrt(crow**2 + ccol**2)
        max_filter_size = max_distance * 1  # 解像度を元に最大のフィルタサイズを設定
        real_filter_size = (1 - filter_size) * max_filter_size
       
        mask = np.zeros((rows, cols), np.uint8)  #　復元するための0＋0iの要素を持つshapeがrows*colsの配列を生成
        for i in range (rows):     #　円内のフィルタリングがされていない部分を判別
            for j in range(cols):
                if (i - crow)**2 + (j - ccol)**2 <= real_filter_size**2:
                    mask[i, j] = 1 

        # 0+0iの部分を復元
        fshift_restored = np.zeros((rows, cols), dtype=complex)
        mask_indices = np.where(mask == 1)
        fshift_restored[mask_indices] = fshift_masked[:len(mask_indices[0])]
        
        f_ishift = np.fft.ifftshift(fshift_restored)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)

        filtered_z_grid = np.zeros_like(z_grid)
        filtered_z_grid[:, :, 0] = img_back
        filtered_z_grid[:, :, 1:] = z_grid[:, :, 1:]

        mask = valid_grid.ravel()
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing='ij')

        # もともと点群があった部分だけを抽出
        points = np.vstack((x_coords.ravel()[mask], y_coords.ravel()[mask], filtered_z_grid[:, :, 0].ravel()[mask])).T
        colors = np.vstack((filtered_z_grid[:, :, 1].ravel()[mask], filtered_z_grid[:, :, 2].ravel()[mask], filtered_z_grid[:, :, 3].ravel()[mask])).T
        
        new_points.append(points)
        new_colors.append(colors)
    
    new_points = np.concatenate(new_points, axis=0)
    new_colors = np.concatenate(new_colors, axis=0)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(new_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(new_colors)

    return filtered_pcd

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
        filter_size = 0.5
        # 現在のフレームの点群に対してローパスフィルタを適用し周波数領域でデータを保存
        fshift_masked_data_reduction, z_grid, valid_grid, x_min, x_max, y_min, y_max = apply_low_pass_filter_to_point_cloud(pcd, height_every_frame, resolution, filter_size)

        fshift_masked_data_reduction_real = np.real(fshift_masked_data_reduction)
        fshift_masked_data_reduction_imag = np.imag(fshift_masked_data_reduction)

        # 修正: 7つの要素に変更
        frequency_domain_data = [
            fshift_masked_data_reduction_real,
            fshift_masked_data_reduction_imag,
            z_grid,
            valid_grid,
            x_min,
            x_max,
            y_min,
            y_max
        ]
        
        frequency_domain_data_list.append(frequency_domain_data) #周波数領域のデータをリストに追加
        
        peripheral_radius = 10 #復元する点群範囲の設定
        
        #近いところの点群だけ、周波数領域から空間領域に変換する
        pcd = regenerate_point_cloud(closest_pose ,peripheral_radius ,filter_size, resolution)
        rospy.loginfo("images_callback: Returned from regenerate_point_cloud")

        camera_positions.append([closest_pose.pose.position.x,
                                 closest_pose.pose.position.y,
                                 closest_pose.pose.position.z])
        marker = create_marker(camera_positions, frame_id)
        arrow_marker = create_arrow_marker(closest_pose, frame_id)

        rospy.loginfo(f"ボクセルダウンサンプリング後の点群には {len(pcd.points)} 点")

        if callback_counter % 1 == 0:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) * 255
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
            marker_pub.publish(arrow_marker)
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
    pose_sub = PoseStampedSubscriber('/home/riku-suzuki/madmax/E-2_ground_truth/gt_6DoF_gnss_and_imu.csv')

    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
    ts.registerCallback(lambda color, depth: images_callback(color, depth, pub, pose_sub, marker_pub, frame_id))

    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception occurred")
