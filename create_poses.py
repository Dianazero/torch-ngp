from audioop import avg
import os
import glob
import numpy as np
import math
import json
import argparse
from scipy.spatial.transform import Slerp, Rotation


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=1):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def ngp_matrix_to_nerf(pose, scale=1):
    new_pose = np.array([
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db): 
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="root directory to the LLFF dataset (contains images/ and pose_bounds.npy)")
    parser.add_argument('--output', type=str, default='images', help="images folder")
    parser.add_argument('--downscale', type=float, default=1, help="image size down scale")

    opt = parser.parse_args()
    

    with open(opt.input) as f:
                transform = json.load(f)
    # load data
    H = int(transform['h']) 
    W = int(transform['w']) 
    fl_x = transform['fl_x']
    fl_y = transform['fl_y']
    cx = transform['cx']
    cy = transform['cy']
    

    frames = transform["frames"]
    frames = sorted(frames, key=lambda d: d['file_path'])  
    
    # choose two random poses, and interpolate between.
    f0, f1 = np.random.choice(frames, 2, replace=False)
    pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=0.1) # [4, 4]
    pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=0.1) # [4, 4]
    rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
    slerp = Slerp([0, 1], rots)

    # construct frames
    frames = []
    for i in range(1, 150+1):
        # ratio = np.sin(((i / self.video_frame_num) - 0.5) * np.pi) * 0.5 + 0.5
        ratio = np.sin(((i/40) - 0.5) * np.pi) * 0.5 + 0.5
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = slerp(ratio).as_matrix()
        pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
        
        pose = ngp_matrix_to_nerf(pose, scale=1) # [4, 4]
        m_path = 'output/chair/' 

        frames.append({
            'file_path':  m_path + f'{i:03d}.jpg',
            'transform_matrix': pose.tolist(),
        })


    # construct a transforms.json
    transforms = {
        'w': W,
        'h': H,
        'fl_x': fl_x,
        'fl_y': fl_y,
        'cx': cx,
        'cy': cy,
        'aabb_scale': 2,
        'frames': frames,
    }

    # write
    output_path = os.path.join(opt.output, 'transforms_new.json')
    print(f'[INFO] write to {output_path}')
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)

