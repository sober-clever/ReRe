CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  sprec.py \
    --base_model model \
    --train_file ./data/Yelp/train/Yelp_5_2021-1-2021-11.csv \
    --eval_file ./data/Yelp/valid/Yelp_5_2021-1-2021-11.csv \
    --info_file ./data/Yelp/info/Yelp_5_2021-1-2021-11.txt \
    --output_dir ./ckpt/exp_name \
    --result_file path_to_result_to_get_self_generated_neg \
    --beta 0.1 \
    --batch_size 128 \
    --micro_batch_size 1 \
    --seed 0 \
    --num_epochs 1 \
    --neg_num 3 \
    --eval_step 0.5 \
    --save_step 0.05 \
    --learning_rate 1e-5 \
    --category Yelp \
    --wandb_project ReRe \
    --wandb_run_name exp_name \
    > ./logs/exp_name.log 2>&1



