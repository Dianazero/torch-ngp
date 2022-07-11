#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python main_dynerf.py ./data/coffee_martini/ --workspace trial_dynerf_5s --preload --epoch 100 --video_frame_num 150  --model_static --model CombineDyNeRFNetwork 
# CUDA_VISIBLE_DEVICES=3 python main_dynerf.py ./data/coffee_martini/ --workspace trial_dynerf_5s --preload --epoch 200 --video_frame_num 150 --error_type isg --model CombineDyNeRFNetwork 

CUDA_VISIBLE_DEVICES=2 python main_dynerf.py ./data/coffee_martini/ --workspace trial_dynerf_5s --preload --test --epoch 200 --video_frame_num 150 --error_type isg --model CombineDyNeRFNetwork --out_video_idx 39 


## nohup sh train_dynerf.sh > run.log 2>&1 &