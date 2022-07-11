import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh
import imageio
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()



def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.mode = opt.mode # colmap, blender, llff
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.sample_mode = opt.sample_mode

        self.video_frame_num = opt.video_frame_num
        self.video_frame_start = opt.video_frame_start
        self.error_map = opt.error_map
        self.times = []
        self.training = self.type in ['train', 'all']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        self.key_frame = opt.key_frame

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])    
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            #f0, f1 = frames[0],frames[-1]
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.times = []
            self.images = None
            n_test = self.video_frame_num
            if self.video_frame_num == 1:
                n_test = 30
            for i in range(1,n_test+1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
                if self.video_frame_num == 1:
                    i = 1
                ts = i*1.0/(n_test+1)
                time = np.array(ts, dtype=np.float32)
                # self.times.append(i*1.0/n_test)
                self.times.append(time)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all': use all frames
                        
            self.poses = []
            self.images = []
            self.times = []
            self.cam_idx = []
            self.isg_map = []
            self.ist_map = []
            self.error_map = []
            for cam_idx,f in tqdm.tqdm(enumerate(frames), desc=f'Loading {type} data:'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and f_path[-4:] != '.png':
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                #print('f:',f)
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
               # print('pose:',pose)
                pose = nerf_matrix_to_ngp(pose, scale=self.scale)

                vid = imageio.get_reader(f_path, 'ffmpeg')
                cam_images = []
                for frame_idx, raw_image in enumerate(vid):

                    if frame_idx < self.video_frame_start:
                        continue
                    if frame_idx >= (self.video_frame_num + self.video_frame_start):
                        break

                    if self.key_frame:
                        if frame_idx % 25 != 0:
                            continue

                    #image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]

                    if self.H is None or self.W is None:
                        self.H = raw_image.shape[0] // downscale
                        self.W = raw_image.shape[1] // downscale

                    # add support for the alpha channel as a mask.
                    # if image.shape[-1] == 3: 
                    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # else:
                    #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                    image = cv2.resize(raw_image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    image = image.astype(np.float32) / 255 # [H, W, 3/4]
                    
                    time = (frame_idx+1)*1.0 / (self.video_frame_num+1)
                    time = np.array(time, dtype=np.float32)
                    self.poses.append(pose)
                    self.images.append(image)
                    self.times.append(time)
                    self.cam_idx.append(cam_idx)
                    cam_images.append(cv2.resize(raw_image, (128, 128)))


                ## calculate ISG weights
                cam_images = np.stack(cam_images, axis=0) # [N,H,W,C]
                isg_map = np.std(cam_images.transpose([1,2,3,0]).reshape([128, 128, -1]), axis=2)
                isg_map = np.power(isg_map/3.0 + 1, np.log(self.video_frame_num)/np.log(10))
                isg_map = isg_map/np.sum(isg_map) # [128, 128]
                self.isg_map.append(isg_map.reshape(-1))
                ## calculate IST weights
                m_cam_images = np.mean(cam_images, axis = 0)
                for im_idx in range(len(cam_images)):
                    isg_map = np.sum(np.abs(cam_images[im_idx] - m_cam_images), axis = -1)
                    isg_map = gaussian_filter(isg_map + 3, sigma=11)
                    isg_map = isg_map/np.sum(isg_map) # [128, 128]
                    self.isg_map.append(isg_map.reshape(-1))

                    # calculate ist map
                    if im_idx == 0:
                        ist_map = np.ones((128,128))
                    else:
                        ist_map = np.sum(np.abs(cam_images[im_idx] - cam_images[im_idx - 1]), axis = -1)
                    ist_map = gaussian_filter(ist_map + 3, sigma=11)
                    ist_map = ist_map/np.sum(ist_map) # [128, 128]
                    self.ist_map.append(ist_map.reshape(-1))

            self.cam_idx = torch.from_numpy(np.array(self.cam_idx))
            self.isg_map = torch.from_numpy(np.stack(self.isg_map, axis=0))
            self.ist_map = torch.from_numpy(np.stack(self.ist_map, axis=0))

        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        self.times = np.array(self.times)
        self.times = torch.from_numpy(np.stack(self.times, axis=0)) # [N, 1]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if type == 'train' and  self.error_map is not None:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None
        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())
        #.to(torch.half if self.fp16 else torch.float)
        if self.preload:
            self.poses = self.poses.to(self.device)
            self.times = self.times.to(self.device)
            if self.images is not None:
                self.images = self.images.to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)
            if self.type == "train":
                self.cam_idx = self.cam_idx.to(self.device)
                if self.sample_mode == "error_map":
                    self.error_map = self.error_map.to(self.device)
                if self.sample_mode == "isg_map":
                    self.isg_map = self.isg_map.to(self.device)
                if self.sample_mode == "ist_map":
                    self.ist_map = self.ist_map.to(self.device)
        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.H / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.W / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):

        B = len(index) # always 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'time': self.times[index].to(self.device),
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        if self.type == "train" and self.sample_mode == "isg_map":
            cam_index = self.cam_idx[index]
            isg_map = self.isg_map[cam_index]
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, isg_map)
        elif self.type == "train" and self.sample_mode == "ist_map":
            ist_map = self.ist_map[index]
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, ist_map)
        elif self.type == "train" and self.sample_mode == "error_map":
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, self.error_map[index])
        else:
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays)


        # error_map = None if self.error_map is None else self.error_map[index]
        
        # rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map)
        
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'time':self.times[index].to(self.device),
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        #if self.sample_mode == "error" and type == 'train':
        if self.error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        return loader
