import argparse
import os
import numpy as np
import json
import cv2
from skvideo import io
from stray_visualize import DEPTH_WIDTH, DEPTH_HEIGHT, _resize_camera_matrix

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1440
OUT_WIDTH = 640
OUT_HEIGHT = 480

from scipy.spatial.transform import Rotation
import itertools
import open3d as o3d

def read_data(path, keyframes=None):
    intrinsics = np.loadtxt(os.path.join(path, 'camera_matrix.csv'), delimiter=',')
    odometry = np.loadtxt(os.path.join(path, 'odometry.csv'), delimiter=',', skiprows=1)
    poses = []

    for i, line in enumerate(odometry):
        if keyframes is None or i in keyframes:
            # x, y, z, qx, qy, qz, qw
            position = line[:3]
            quaternion = line[3:]
            T_WC = np.eye(4)
            T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            T_WC[:3, 3] = position
            poses.append(T_WC)
    return { 'poses': poses, 'intrinsics': intrinsics }

def write_params(flags):
# def stray_to_o3d(path, poses, intrinsic_matrix, width, height, indexes=itertools.count()):
    indexes=itertools.count()

    poses, intrinsics = read_data(flags.dataset).values()
    intrinsics_scaled = _resize_camera_matrix(intrinsics, OUT_WIDTH / FRAME_WIDTH, OUT_HEIGHT / FRAME_HEIGHT)

    for i, pose in zip(indexes, poses):
        params = o3d.camera.PinholeCameraParameters()
        params.extrinsic = np.linalg.inv(pose)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=OUT_WIDTH, height=OUT_HEIGHT, 
                                                          fx=intrinsics_scaled[0,0], 
                                                          fy=intrinsics_scaled[1,1],
                                                          cx=intrinsics_scaled[0,2],
                                                          cy=intrinsics_scaled[1,2],
                                                         )
        params.intrinsic = intrinsic

        o3d.io.write_pinhole_camera_parameters(os.path.join(flags.out, f"{i:06}.o3d.json"), params)
        write_frame_json(os.path.join(flags.out, f"Frame.{i:04}.jpg.json"), params, i)

def write_frame_json(path, params, frame_number):
    with open(path, 'w') as f:
        json.dump({
            'frameNum': frame_number,
            'cameraPosition': dict(zip('xyz', params.extrinsic[:3, 3])),
            'focalLength': dict(zip('xy', params.intrinsic.get_focal_length())),
            'principalPoint': dict(zip('xy', params.intrinsic.get_principal_point())),
            'resolution': dict(zip('xy', (params.intrinsic.width, params.intrinsic.height))),
            'transform': params.extrinsic.tolist()
            }, 
        f, indent=2)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--confidence', type=int, default=2)
    return parser.parse_args()

def write_frames(flags, rgb_out_dir):
    rgb_video = os.path.join(flags.dataset, 'rgb.mp4')
    video = io.vreader(rgb_video)
    for i, frame in enumerate(video):
        print(f"Writing rgb frame {i:06}" + " " * 10, end='\r')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT))
        frame_path = os.path.join(rgb_out_dir, f"{i:06}.jpg")
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        cv2.imwrite(frame_path, frame, params)

def resize_depth(depth):
    out = cv2.resize(depth, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_NEAREST_EXACT)
    out[out < 10] = 0
    return out

def write_depth(flags, depth_out_dir):
    depth_dir_in = os.path.join(flags.dataset, 'depth')
    confidence_dir = os.path.join(flags.dataset, 'confidence')
    files = sorted(os.listdir(depth_dir_in))
    for filename in files:
        if '.npy' not in filename:
            continue
        print(f"Writing depth frame {filename}", end='\r')
        number, _ = filename.split('.')
        depth = np.load(os.path.join(depth_dir_in, filename))
        confidence = cv2.imread(os.path.join(confidence_dir, number + '.png'))[:, :, 0]
        depth[confidence < flags.confidence] = 0
        depth = resize_depth(depth)
        cv2.imwrite(os.path.join(depth_out_dir, number + '.png'), depth)

def write_intrinsics(flags):
    intrinsics = np.loadtxt(os.path.join(flags.dataset, 'camera_matrix.csv'), delimiter=',')
    data = {}
    intrinsics_scaled = _resize_camera_matrix(intrinsics, OUT_WIDTH / FRAME_WIDTH, OUT_HEIGHT / FRAME_HEIGHT)
    data['intrinsic_matrix'] = [intrinsics_scaled[0, 0], 0.0, 0.0,
            0.0, intrinsics_scaled[1, 1], 0.0,
            intrinsics_scaled[0, 2], intrinsics_scaled[1, 2], 1.0]
    data['width'] = OUT_WIDTH
    data['height'] = OUT_HEIGHT
    data['depth_scale'] = 1000.0
    data['fps'] = 60.0
    data['depth_format'] = 'Z16'
    with open(os.path.join(flags.out, 'camera_intrinsics.json'), 'wt') as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))

def write_config(flags):
    dataset_path = os.path.abspath(os.path.expanduser(flags.out))
    intrinsics_path = os.path.join(dataset_path, 'camera_intrinsics.json')
    config = {
        "name": "Stray Scanner dataset",
        "path_dataset": dataset_path,
        "path_intrinsic": intrinsics_path,
        "depth_scale": 1000.0,
        "max_depth": 10.0,
        "min_depth": 0.05,
        "python_multi_threading": False
    }
    with open(os.path.join(dataset_path, 'config.json'), 'w') as f:
        f.write(json.dumps(config, indent=4, sort_keys=True))

def main():
    flags = read_args()
    rgb_out = os.path.join(flags.out, 'color/')
    depth_out = os.path.join(flags.out, 'depth/')
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)

    write_params(flags)
    write_config(flags)
    write_intrinsics(flags)
    write_depth(flags, depth_out)
    write_frames(flags, rgb_out)
    print("\nDone.")



if __name__ == "__main__":
    main()

