import fire
import os
import pandas as pd

def split(input_path, output_path, cuda_list):
    if type(cuda_list) == int:
        cuda_list = [cuda_list]
    df = pd.read_csv(input_path)
    # df = df.sample(frac=1).reset_index(drop=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_len = len(df)
    cuda_list = list(cuda_list)
    cuda_num = len(cuda_list)
    for i in range(cuda_num):
        start = i * df_len // cuda_num
        end = (i+1) * df_len // cuda_num
        df[start:end].to_csv(f'{output_path}/{cuda_list[i]}.csv', index=True)
        
if __name__ == '__main__':
    fire.Fire(split)
