from audioop import avg
import os
import glob
import numpy as np
import math
import json
import cv2
import argparse
import imageio


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="root directory to the LLFF dataset (contains images/ and pose_bounds.npy)")
    parser.add_argument('--images', type=str, default='images', help="images folder")
    parser.add_argument('--output', type=str, default='images', help="images folder")
    parser.add_argument('--downscale', type=float, default=1, help="image size down scale")

    opt = parser.parse_args()
    print(f'[INFO] process {opt.path}')
    
    # load data
    images = [f[len(opt.path):] for f in sorted(glob.glob(os.path.join(opt.path, opt.images,"*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg') or f.lower().endswith('mp4')]
    
    poses_bounds = np.load(os.path.join(opt.path, 'poses_bounds.npy'))
    N = poses_bounds.shape[0]

    print(f'[INFO] loaded {len(images)} images, {N} poses_bounds as {poses_bounds.shape}')
    frame_count = 0
    for f in images:
        filename = opt.path + f
        cap = cv2.VideoCapture(filename)
        # vid = imageio.get_reader(filename, 'ffmpeg') 
        success = True
        while(success): 
            success, frame = cap.read()
            out_image = opt.output + str(frame_count) + '.png'
            cv2.imwrite(out_image,frame)
            frame_count += 1
            break
        # for num,im in enumerate(vid): 
        #     out_image = opt.output + str(num) + '.png'
        #     cv2.imwrite(out_image,im)
        #     break





