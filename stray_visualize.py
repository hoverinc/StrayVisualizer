import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
import skvideo.io

import multiprocessing
from multiprocessing.pool import ThreadPool
from functools import partial

description = """
This script visualizes datasets collected using the Stray Scanner app.
"""

usage = """
Basic usage: python stray_visualize.py <path-to-dataset-folder>
"""

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('path', type=str, help="Path to StrayScanner dataset to process.")
    parser.add_argument('--trajectory', '-t', action='store_true', help="Visualize the trajectory of the camera as a line.")
    parser.add_argument('--frames', '-f', action='store_true', help="Visualize camera coordinate frames from the odometry file.")
    parser.add_argument('--point-clouds', '-p', action='store_true', help="Show concatenated point clouds.")
    parser.add_argument('--color-point-clouds', action='store_true', help="Show concatenated color point clouds.")
    parser.add_argument('--integrate', '-i', action='store_true', help="Integrate point clouds using the Open3D RGB-D integration pipeline, and visualize it.")
    parser.add_argument('--mesh-filename', type=str, help='Mesh generated from point cloud integration will be stored in this file. open3d.io.write_triangle_mesh will be used.', default=None)
    parser.add_argument('--every', type=int, default=60, help="Show only every nth point cloud and coordinate frames. Only used for point cloud and odometry visualization.")
    parser.add_argument('--voxel-size', type=float, default=0.015, help="Voxel size in meters to use in RGB-D integration.")
    # parser.add_argument('--depth-width', type=int, default=256, help="Width RGB-D Images are scaled to")
    # parser.add_argument('--depth-height', type=int, default=192, help="height RGB-D Images are scaled to")
    parser.add_argument('--confidence', '-c', type=int, default=1,
            help="Keep only depth estimates with confidence equal or higher to the given value. There are three different levels: 0, 1 and 2. Higher is more confident.")
    parser.add_argument('--integrate-every', type=int, default=1, help="Only integrate every nth frame.")
    return parser.parse_args()

def _resize_camera_matrix(camera_matrix, scale_x, scale_y):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return np.array([[fx * scale_x, 0.0, cx * scale_x],
        [0., fy * scale_y, cy * scale_y],
        [0., 0., 1.0]])

def read_data(flags):
    intrinsics = np.loadtxt(os.path.join(flags.path, 'camera_matrix.csv'), delimiter=',')
    odometry = np.loadtxt(os.path.join(flags.path, 'odometry.csv'), delimiter=',', skiprows=1)
    poses = []

    for line in odometry:
        # x, y, z, qx, qy, qz, qw
        position = line[:3]
        quaternion = line[3:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        T_WC[:3, 3] = position
        poses.append(T_WC)
    return { 'poses': poses, 'intrinsics': intrinsics }

def load_depth(path, confidence=None, filter_level=0):
    depth_mm = np.load(path)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    return o3d.geometry.Image(depth_m)

def load_confidence(path):
    return np.array(Image.open(path))

def get_intrinsics(intrinsics):
    """
    Scales the intrinsics matrix to be of the appropriate scale for the depth maps.
    """
    intrinsics_scaled = _resize_camera_matrix(intrinsics, DEPTH_WIDTH / 1920, DEPTH_HEIGHT / 1440)
    return o3d.camera.PinholeCameraIntrinsic(width=DEPTH_WIDTH, height=DEPTH_HEIGHT, fx=intrinsics_scaled[0, 0],
        fy=intrinsics_scaled[1, 1], cx=intrinsics_scaled[0, 2], cy=intrinsics_scaled[1, 2])

def trajectory(flags, data):
    """
    Returns a set of lines connecting each camera poses world frame position.
    returns: [open3d.geometry.LineSet]
    """
    line_sets = []
    previous_pose = None
    for i, T_WC in enumerate(data['poses']):
        if previous_pose is not None:
            points = o3d.utility.Vector3dVector([previous_pose[:3, 3], T_WC[:3, 3]])
            lines = o3d.utility.Vector2iVector([[0, 1]])
            line = o3d.geometry.LineSet(points=points, lines=lines)
            line_sets.append(line)
        previous_pose = T_WC
    return line_sets

def show_frames(flags, data):
    """
    Returns a list of meshes of coordinate axes that have been transformed to represent the camera matrix
    at each --every:th frame.

    flags: Command line arguments
    data: dict with keys ['poses', 'intrinsics']
    returns: [open3d.geometry.TriangleMesh]
    """
    frames = [o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.25, np.zeros(3))]
    for i, T_WC in enumerate(data['poses']):
        if not i % flags.every == 0:
            continue
        print(f"Frame {i}", end="\r")
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, np.zeros(3))
        frames.append(mesh.transform(T_WC))
    return frames

def point_clouds(flags, data):
    """
    Converts depth maps to point clouds and merges them all into one global point cloud.
    flags: command line arguments
    data: dict with keys ['intrinsics', 'poses']
    returns: [open3d.geometry.PointCloud]
    """
    pcs = []
    intrinsics = get_intrinsics(data['intrinsics'])
    pc = o3d.geometry.PointCloud()
    meshes = []
    for i, T_WC in enumerate(data['poses']):
        if i % flags.every != 0:
            continue
        print(f"Point cloud {i}", end="\r")
        T_CW = np.linalg.inv(T_WC)
        confidence = load_confidence(os.path.join(flags.path, 'confidence', f'{i:06}.png'))
        depth = load_depth(os.path.join(flags.path, 'depth', f'{i:06}.npy'), confidence, filter_level=flags.confidence)
        
        pc += o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsic=T_CW, depth_scale=1.0)
    return [pc]

# def color_point_clouds(flags, data):
#     intrinsics = get_intrinsics(data['intrinsics'])
#     pc = o3d.geometry.PointCloud()

#     rgb_path = os.path.join(flags.path, 'rgb.mp4')
#     video = skvideo.io.vreader(rgb_path)
#     for i, (T_WC, rgb) in enumerate(zip(data['poses'], video)):
#         depth_path = os.path.join(flags.path, 'depth', f'{i:06}.npy')
#         try:
#             depth = load_depth(depth_path)
#         except FileNotFoundError:
#             print(f"Missing/Skipping frame {i:06}", end='\r')
#             continue
#         rgb = Image.fromarray(rgb)
#         rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
#         rgb = np.array(rgb)
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             o3d.geometry.Image(rgb), depth,
#             depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False)

#         print(f"Point cloud {i}", end="\r")
#         T_CW = np.linalg.inv(T_WC)
#         # pc += o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsic=T_CW, depth_scale=1.0)
#         pc += o3d.geometry.PointCloud.create_from_rgbd_image(
#             image=rgbd, intrinsic=intrinsics, extrinsic=T_CW)

#     return [pc]

def color_point_clouds(flags, data):
    def preload(inputs):
        i, (T_WC, rgb) = inputs
        depth_path = os.path.join(flags.path, 'depth', f'{i:06}.npy')
        try:
            depth = load_depth(depth_path)
        except FileNotFoundError:
            print(f"Missing/Skipping frame {i:06}", end='\r')
            return i, (T_WC, None, None)
        rgb = Image.fromarray(rgb)
        rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb = np.array(rgb)
        rgb = o3d.geometry.Image(rgb)

        return i, (T_WC, rgb, depth)

    intrinsics = get_intrinsics(data['intrinsics'])
    pc = o3d.geometry.PointCloud()

    rgb_path = os.path.join(flags.path, 'rgb.mp4')
    video = skvideo.io.vreader(rgb_path)
    with ThreadPool() as pool:
        for i, (T_WC, rgb, depth) in pool.imap_unordered(preload, enumerate(zip(data['poses'], video))): 
            if depth is None: continue         
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, depth,
                depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False)

            print(f"Point cloud {i}", end="\r")
            T_CW = np.linalg.inv(T_WC)
            # pc += o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsic=T_CW, depth_scale=1.0)
            pc += o3d.geometry.PointCloud.create_from_rgbd_image(
                image=rgbd, intrinsic=intrinsics, extrinsic=T_CW)

    return [pc]

def integrate_frame(T_WC, rgb, volume):
    print(f"Integrating frame {i:06}     ", end='\r')
    depth_path = os.path.join(flags.path, 'depth', f'{i:06}.npy')
    try:
        depth = load_depth(depth_path)
    except FileNotFoundError:
        print(f"Missing/Skipping frame {i:06}", end='\r')
        return
    rgb = Image.fromarray(rgb)
    rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
    rgb = np.array(rgb)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb), depth,
        depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False)

    volume.integrate(rgbd, intrinsics, np.linalg.inv(T_WC))

def integrate_frame_wrapper(args, volume, integrate_every=1):
    i, (T_WC, rgb) = args
    if i % integrate_every != 0: 
        return integrate_frame(T_WC, rgb, volume)

def integrate(flags, data, integrate_every=1, num_threads=128):
    """
    Integrates collected RGB-D maps using the Open3D integration pipeline.

    flags: command line arguments
    data: dict with keys ['intrinsics', 'poses']
    Returns: open3d.geometry.TriangleMesh
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=flags.voxel_size,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intrinsics = get_intrinsics(data['intrinsics'])

    rgb_path = os.path.join(flags.path, 'rgb.mp4')
    video = skvideo.io.vreader(rgb_path)
    f = partial(integrate_frame_wrapper, volume=volume, integrate_every=integrate_every)
    with ThreadPool(num_threads) as pool:
        for _ in pool.imap_unordered(f, enumerate(zip(data['poses'], video))):
            pass
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

def integrate_old(flags, data, integrate_every=1):
    """
    Integrates collected RGB-D maps using the Open3D integration pipeline.

    flags: command line arguments
    data: dict with keys ['intrinsics', 'poses']
    Returns: open3d.geometry.TriangleMesh
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=flags.voxel_size,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intrinsics = get_intrinsics(data['intrinsics'])

    rgb_path = os.path.join(flags.path, 'rgb.mp4')
    video = skvideo.io.vreader(rgb_path)
    for i, (T_WC, rgb) in enumerate(zip(data['poses'], video)):
        if i % integrate_every != 0: continue
        print(f"Integrating frame {i:06}     ", end='\r')
        depth_path = os.path.join(flags.path, 'depth', f'{i:06}.npy')
        try:
            depth = load_depth(depth_path)
        except FileNotFoundError:
            print(f"Missing/Skipping frame {i:06}", end='\r')
            continue
        rgb = Image.fromarray(rgb)
        rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb = np.array(rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), depth,
            depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False)

        volume.integrate(rgbd, intrinsics, np.linalg.inv(T_WC))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def validate(flags):
    if not os.path.exists(os.path.join(flags.path, 'rgb.mp4')):
        absolute_path = os.path.abspath(flags.path)
        print(f"The directory {absolute_path} does not appear to be a directory created by the Stray Scanner app.")
        return False
    return True

def main():
    flags = read_args()

    if not validate(flags):
        return

    if not flags.frames and not flags.point_clouds and not flags.integrate and not flags.color_point_clouds:
        flags.frames = True
        flags.point_clouds = True
        flags.trajectory = True

    data = read_data(flags)
    geometries = []
    if flags.trajectory:
        geometries += trajectory(flags, data)
    if flags.frames:
        geometries += show_frames(flags, data)
    if flags.point_clouds:
        geometries += point_clouds(flags, data)
    if flags.color_point_clouds:
        geometries += color_point_clouds(flags, data)
    if flags.integrate:
        print(flags)
        mesh = integrate(flags, data, flags.integrate_every)
        if flags.mesh_filename is not None:
            o3d.io.write_triangle_mesh(flags.mesh_filename, mesh)
        geometries += [mesh]
    # o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    main()

