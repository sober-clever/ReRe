for category in "Industrial_and_Scientific"
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${test_fie} ${info_file} ${eval_file}
    
    torchrun --nproc_per_node 4 \
            sft.py \
            --base_model path_to_model \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir \
            --wandb_project wandb_proj \
            --wandb_run_name wandb_name \
            --category ${category} \
            --train_from_scratch False\
            --seed 42
    
done

