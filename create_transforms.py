#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

from scipy.spatial.transform import Slerp, Rotation



def parse_args():
	parser = argparse.ArgumentParser(description="convert to a nerf style transforms.json from which to save screenshots.")

	parser.add_argument("--input", default="", help="trained transforms.json to be convert.")
	parser.add_argument('--n_tests', type=int, default=30, help="number of test frames.")
	parser.add_argument("--output", default="transforms.json", help="output path")
	args = parser.parse_args()
	return args


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
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


# Rather ugly pose generation code, derived from NeRF
def _trans_t(t):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rot_phi(phi):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rot_theta(th):
    return np.array(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

def pose_spherical(theta : float, phi : float, radius : float, offset : np.ndarray=None,
                   vec_up : np.ndarray=None):
    """
    Generate spherical rendering poses, from NeRF. Forgive the code horror
    :return: r (3,), t (3,)
    """
    c2w = _trans_t(radius)
    c2w = _rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = _rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        np.array(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        @ c2w
    )
    if vec_up is not None:
        vec_up = vec_up / np.linalg.norm(vec_up)
        vec_1 = np.array([vec_up[0], -vec_up[2], vec_up[1]])
        vec_2 = np.cross(vec_up, vec_1)

        trans = np.eye(4, 4, dtype=np.float32)
        trans[:3, 0] = vec_1
        trans[:3, 1] = vec_2
        trans[:3, 2] = vec_up
        c2w = trans @ c2w
    c2w = c2w @ np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    if offset is not None:
        c2w[:3, 3] += offset
    return c2w


if __name__ == "__main__":
	args = parse_args()
	IN_PATH = args.input + 'transforms.json'
	OUT_PATH = args.output + 'transforms.json'
	print(f"outputting to {OUT_PATH}...")
	with open(IN_PATH, 'r') as f:
		transform = json.load(f)

	frames = transform["frames"]
	frames = sorted(frames, key=lambda d: d['file_path'])
	# f0, f1 = np.random.choice(frames, 2, replace=False)
	f0, f1, f2 ,f3 = frames[0], frames[10], frames[26], frames[37]
	# for i in range(len(frames)):
		# print('i: ',i,frames[i]["file_path"])
	pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=1) # [4, 4]
	pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=1) # [4, 4]
	pose2 = nerf_matrix_to_ngp(np.array(f2['transform_matrix'], dtype=np.float32), scale=1) # [4, 4]
	pose3 = nerf_matrix_to_ngp(np.array(f3['transform_matrix'], dtype=np.float32), scale=1) # [4, 4]

	rots0 = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
	slerp0 = Slerp([0, 1], rots0)
	rots1 = Rotation.from_matrix(np.stack([pose1[:3, :3], pose2[:3, :3]]))
	slerp1 = Slerp([0, 1], rots1)
	rots2 = Rotation.from_matrix(np.stack([pose2[:3, :3], pose3[:3, :3]]))
	slerp2 = Slerp([0, 1], rots2)
	rots3 = Rotation.from_matrix(np.stack([pose3[:3, :3], pose0[:3, :3]]))
	slerp3 = Slerp([0, 1], rots3)

	print(len(frames))

	n_test = 120  # args.n_tests
	new_frames = []
	for i in range(n_test):
		if i<30:
			ratio = np.sin(((i/30) - 0.5) * np.pi) * 0.5 + 0.5
			pose = np.eye(4, dtype=np.float32)
			pose[:3, :3] = slerp0(ratio).as_matrix()
			pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
			pose = ngp_matrix_to_nerf(pose)
		elif i<60:
			ratio = np.sin((((i-30) / 30) - 0.5) * np.pi) * 0.5 + 0.5
			pose = np.eye(4, dtype=np.float32)
			pose[:3, :3] = slerp1(ratio).as_matrix()
			pose[:3, 3] = (1 - ratio) * pose1[:3, 3] + ratio * pose2[:3, 3]
			pose = ngp_matrix_to_nerf(pose)
		elif i<90:
			ratio = np.sin((((i-60) / 30) - 0.5) * np.pi) * 0.5 + 0.5
			pose = np.eye(4, dtype=np.float32)
			pose[:3, :3] = slerp2(ratio).as_matrix()
			pose[:3, 3] = (1 - ratio) * pose2[:3, 3] + ratio * pose3[:3, 3]
			pose = ngp_matrix_to_nerf(pose)
		else:
			ratio = np.sin((((i-90) / 30) - 0.5) * np.pi) * 0.5 + 0.5
			pose = np.eye(4, dtype=np.float32)
			pose[:3, :3] = slerp3(ratio).as_matrix()
			pose[:3, 3] = (1 - ratio) * pose3[:3, 3] + ratio * pose0[:3, 3]
			pose = ngp_matrix_to_nerf(pose)
		name = args.output + str(i) + '.png'
		frame={"file_path":name,"transform_matrix": pose}
		new_frames.append(frame)


		
	# up_rot = dset.c2w[:, :3, :3].cpu().numpy()
	# ups = np.matmul(up_rot, np.array([0, -1.0, 0])[None, :, None])[..., 0]
	# vec_up = np.mean(ups, axis=0)
	# vec_up /= np.linalg.norm(vec_up)
	# print('  Auto vec_up', vec_up)
	# vec_up = None

	# num_views = 10
	# elevation = -12.0
	# elevation2 = -12.0
	# radius = 0.5
	# offset= 0,0,0

	# angles = np.linspace(-180, 180, num_views + 1)[:-1]
	# elevations = np.linspace(elevation, elevation2, num_views)

	# i=0
	# for ele, angle in zip(elevations, angles):
	# 	pose = pose_spherical(
    #         angle,
    #         ele,
    #         radius,
    #         offset,
    #         vec_up=vec_up,
    #     )
	# 	# pose = ngp_matrix_to_nerf(pose)
	# 	pose[2,:] *= -1 # flip whole world upside down
	# 	name = args.output + str(i) + '.png'
	# 	frame={"file_path":name,"transform_matrix": pose}
	# 	new_frames.append(frame)
	# 	i+=1

	# c2ws = [
    #     pose_spherical(
    #         angle,
    #         ele,
    #         radius,
    #         offset,
    #         vec_up=vec_up,
    #     )
    #     for ele, angle in zip(elevations, angles)
    # ]
	# c2ws += [
    #     pose_spherical(
    #         angle,
    #         ele,
    #         radius,
    #         offset,
    #         vec_up=vec_up,
    #     )
    #     for ele, angle in zip(reversed(elevations), angles)
    # ]
	# print('c2ws:',c2ws)

	# print(new_frames)
	transform["frames"] = new_frames
	for f in transform["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()
	with open(OUT_PATH, "w") as outfile:
		json.dump(transform, outfile, indent=2)
