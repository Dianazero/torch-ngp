#!/bin/bash

#CUDA_VISIBLE_DEVICES=3 python main_nerf.py data/workspace_TV/ --workspace trial_time -O  --scale 0.2
# CUDA_VISIBLE_DEVICES=0 python main_nerf.py  ./data/coffee_martini/  --workspace trial_tv_multires10_256_128 --cuda_ray --fp16   --scale 0.12  --video_frame_num 180   --max_epoch 100 --tv_loss
# CUDA_VISIBLE_DEVICES=3 python main_nerf.py  ./data/coffee_martini/  --workspace trial_tv_multires10_256_128 --cuda_ray --fp16   --scale 0.12  --video_frame_num 180   --max_epoch 300  --tv_loss --sample_mode ist_map #--W 2704 --H 2028

# CUDA_VISIBLE_DEVICES=0 python main_nerf.py ./data/cook_spinach/  --workspace trial_ngp2 --cuda_ray --fp16   --scale 0.12  --video_frame_num 150   --test

# CUDA_VISIBLE_DEVICES=0 python main_nerf.py ./data/cook_spinach/  --workspace trial_time_0 --cuda_ray  --mlp   --scale 0.1  --test

# CUDA_VISIBLE_DEVICES=3 python main_ngp_dynerf.py  ./data/coffee_martini/  --workspace trial_ngp_dynerf  --cuda_ray --fp16 --preload --scale 0.12  --video_frame_num 20   --max_epoch 30 --lr 1e-2 --sample_mode ist_map
# CUDA_VISIBLE_DEVICES=3 python main_nerf_finetune.py  ./data/coffee_martini/  --workspace trial_test_finetune -O --scale 0.12  --video_frame_num 10   --max_epoch 1  --sample_mode ist_map --finetune
# CUDA_VISIBLE_DEVICES=3 python main_nerf.py  ./data/coffee_martini/  --workspace trial_tv_test --cuda_ray --fp16   --scale 0.12  --video_frame_num 5   --max_epoch 150 --sample_mode ist_map

# CUDA_VISIBLE_DEVICES=3 python main_nerf.py  ./data/coffee_martini/  --workspace trial_tv_multires12_64_64 --cuda_ray --fp16 --preload  --scale 0.12  --video_frame_num 30   --max_epoch 50  --tv_loss --sample_mode ist_map #--W 2704 --H 2028
# CUDA_VISIBLE_DEVICES=3 python main_nerf.py  ./data/coffee_martini/  --workspace trial_mul_1_256_256 --cuda_ray --fp16 --mlp  --scale 0.12  --video_frame_num 120  --max_epoch 100  --sample_mode isg_map # --tv_loss --tv_loss_weight 1e-5 #--W 2704 --H 2028
#
# CUDA_VISIBLE_DEVICES=3 python main_nerf.py  ./data/coffee_martini/  --workspace trial_6s_1_256_256 --cuda_ray --fp16 --preload --mlp --scale 0.12  --video_frame_num 50  --max_epoch 200  --sample_mode isg_map # --tv_loss --tv_loss_weight 1e-5 #--W 2704 --H 2028
# CUDA_VISIBLE_DEVICES=3 python main_nerf.py  ./data/coffee_martini/  --workspace trial_5s_ngp  --scale 0.12  --video_frame_num 121 --max_epoch 1000 --mlp  --lr 1e-4 --key_frame
# CUDA_VISIBLE_DEVICES=3 python main_nerf.py  ./data/coffee_martini/  --workspace trial_5s_ngp  --scale 0.12  --video_frame_num 121 --max_epoch 1100 --mlp  --lr 1e-5 --sample_mode ist_map
CUDA_VISIBLE_DEVICES=3 python main_nerf.py  ./data/coffee_martini/  --workspace trial_5s_ngp  --scale 0.12  --video_frame_num 121 --max_epoch 1150 --mlp  --lr 1e-5 --sample_mode isg_map

# CUDA_VISIBLE_DEVICES=0 python main_nerf.py  ./data/coffee_martini/  --workspace trial_1s_256_256  --cuda_ray --fp16 --preload --mlp --scale 0.12  --video_frame_num 21 --max_epoch 100 
# CUDA_VISIBLE_DEVICES=0 python main_nerf.py  ./data/coffee_martini/  --workspace trial_1s_256_256  --cuda_ray --fp16 --preload --mlp --scale 0.12  --video_frame_num 21 --max_epoch 300 --sample_mode isg_map


## bash scripts/install_ext.sh
## nohup sh test.sh > run.log 2>&1 &
