for category in "Toys_and_Games"
do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes 4 --main_process_port 29503 \
                                    rere.py \
                                    --model_path path_to_model \
                                    --train_batch_size 64 \
                                    --eval_batch_size 128 \
                                    --gradient_accumulation_steps 2 \
                                    --train_file ${train_file} \
                                    --eval_file ${eval_file} \
                                    --info_file ${info_file} \
                                    --category ${category} \
                                    --sample_frac 1.0 \
                                    --sample_train False \
                                    --eval_step 0.5 \
                                    --reward_type ranking \
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
                                    --output_dir output_dir \
                                    --wandb_run_name wandb_name
done





