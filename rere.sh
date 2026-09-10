export WANDB_MODE="offline"
{
for category in "Yelp"
do
    train_file=$(ls -f ./data/Yelp/train/${category}*.csv)
    eval_file=$(ls -f ./data/Yelp/valid/${category}*.csv)
    info_file=$(ls -f ./data/Yelp/info/${category}*.txt)

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes 8 --main_process_port 29503 \
                                    rere.py \
                                    --model_path path_to_model \
                                    --train_batch_size 64 \
                                    --eval_batch_size 128 \
                                    --gradient_accumulation_steps 1 \
                                    --train_file ${train_file} \
                                    --eval_file ${eval_file} \
                                    --info_file ${info_file} \
                                    --category ${category} \
                                    --sample_train False \
                                    --eval_step 0.5 \
                                    --reward_type ranking \
                                    --seed 0 \
                                    --num_generations 16 \
                                    --num_train_epochs 2 \
                                    --mask_all_zero False \
                                    --dynamic_sampling False \
                                    --sync_ref_model True \
                                    --beam_search True \
                                    --test_during_training False \
                                    --temperature 1.0 \
                                    --learning_rate 1e-5 \
                                    --add_gt False \
                                    --beta 1e-3 \
                                    --dapo False \
                                    --gspo True \
                                    --output_dir ./ckpt/exp_name \
                                    --wandb_run_name exp_name
done
} > logs/exp_name.log 2>&1
