for category in "Toys_and_Games"
do
    exp_name="ckpt_path"
    exp_name_clean=$(basename "$(dirname "$exp_name")")
    file=$(ls -f ./data/Amazon/train/${category}*.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    python3 ./split.py --input_path ${test_file} --output_path ./temp/${category}-$exp_name_clean --cuda_list "0,1,2,3,4,5,6,7"
    cudalist="0 1 2 3 4 5 6 7"
    for i in ${cudalist}
    do
        echo $i
        CUDA_VISIBLE_DEVICES=$i python3 -u ./evaluate.py --base_model $exp_name --info_file ${info_file} --category ${category} --num_beams 50 --batch_size 4 --length_penalty 1.0\
         --test_data_path ./temp/${category}-$exp_name_clean/${i}.csv --result_json_data ./temp/${category}-$exp_name_clean/${i}.json &
    done
    wait
    python3 ./merge.py --input_path ./temp/${category}-$exp_name_clean --output_path $exp_name/final_result.json --cuda_list "0,1,2,3,4,5,6,7"
    python3 ./calc.py --path  $exp_name/final_result.json --item_path ${info_file} > test_file 2>&1
done

