export WANDB_MODE="offline"

{
for category in "Yelp"
do
    train_file=$(ls -f ./data/Yelp/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Yelp/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Yelp/test/${category}*11.csv)
    info_file=$(ls -f ./data/Yelp/info/${category}*.txt)
    echo ${train_file} ${test_fie} ${info_file} ${eval_file}
    
    torchrun --nproc_per_node 8 \
            sft.py \
            --base_model path_to_model \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ./ckpt/exp_name \
            --wandb_project ReRe \
            --wandb_run_name exp_name \
            --category ${category} \
            --train_from_scratch False\
            --num_epochs 10 \
            --seed 100
done
} > logs/exp_name.log 2>&1 