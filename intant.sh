
#!/bin/bash



CUDA_VISIBLE_DEVICES=3 python main_dynerf.py ./data/coffee_martini/ --workspace trial_dynerf_4s --cuda_ray --fp16 --downscale 1 --epoch 25 --video_frame_num 120  --model_static --model CombineDyNeRFNetwork --eval_interval 50
CUDA_VISIBLE_DEVICES=3 python main_dynerf.py ./data/coffee_martini/ --workspace trial_dynerf_4s --cuda_ray --fp16 --downscale 1 --epoch 100 --video_frame_num 120 --error_type isg --model CombineDyNeRFNetwork 

# CUDA_VISIBLE_DEVICES=1 python main_nerf.py  ./data/coffee_martini/  --workspace trial_ngp_sigam_tv  --preload --cuda_ray --fp16  --scale 0.12  --video_frame_num 20 --max_epoch 200 --lr 5e-3 --sample_mode isg_map  --dt_gamma 0 # --tv_loss --tv_loss_weight 1e-6 

# CUDA_VISIBLE_DEVICES=0 python main_nerf.py  ./data/coffee_martini/  --workspace trial_ngp_tv  --cuda_ray --fp16  --scale 0.12  --video_frame_num 61 --max_epoch 300 --lr 5e-3 --sample_mode isg_map  --tv_loss --tv_loss_weight 1e-6

# CUDA_VISIBLE_DEVICES=0 python main_tensoRF.py  ./data/coffee_martini/  --workspace trial_tensoRF --scale 0.12 

## nohup sh intant.sh > run_ngp.log 2>&1 