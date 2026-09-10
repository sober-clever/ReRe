export WANDB_MODE="offline"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  sdpo.py \
    --base_model ./ckpt/sft_yelp_seed0_v3 \
    --train_file ./data/Yelp_timesplit/train/Yelp_5_2021-time-split-11.csv \
    --eval_file ./data/Yelp_timesplit/valid/Yelp_5_2021-time-split-11.csv \
    --info_file ./data/Yelp_timesplit/info/Yelp_5_2021-time-split-11.txt \
    --output_dir ./debug \
    --beta 0.1 \
    --batch_size 128 \
    --micro_batch_size 1 \
    --seed 0 \
    --num_epochs 3\
    --neg_num 3 \
    --eval_step 0.1 \
    --learning_rate 1e-5 \
    --category Yelp \
    --wandb_project MiniOneRec \
    --wandb_run_name debug \
    > ./logs/debug.log 2>&1

