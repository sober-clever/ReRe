for category in "Toys_and_Games"
do
    exp_name= "path_to_model"
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    python ./split.py --input_path ${test_file} --output_path ./temp/${category}-$exp_name --cuda_list "0,1,2,3,4,5,6,7"
    cudalist="0 1 2 3 4 5 6 7"
    for i in ${cudalist}
    do
        echo $i
        CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py --base_model $exp_name --info_file ${info_file} --category ${category} --test_data_path ./temp/${category}-$exp_name/${i}.csv --result_json_data ./temp/${category}-$exp_name/${i}.json &
    done
    wait
    python ./merge.py --input_path ./temp/${category}-$exp_name --output_path $exp_name/final_result.json --cuda_list "0,1,2,3,4,5,6,7"
    python ./calc.py --path  $exp_name/final_result.json --item_path ${info_file} 
done



