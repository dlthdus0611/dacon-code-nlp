#%%
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from rank_bm25 import BM25Okapi
from itertools import combinations
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
#%%
def sampling_func(data, sample_pct):
    np.random.seed(42)
    N = len(data)
    sample_n = int(len(data)*sample_pct) # integer
    sample = data.take(np.random.permutation(N)[:sample_n])
    return sample

def preprocess(script):
    with open(script,'r',encoding='utf-8') as file:
        lines = file.readlines()
        preproc_lines = []
        for line in lines:
            if line.lstrip().startswith('#'):
                continue
            line = line.rstrip()
            if '#' in line:
                line = line[:line.index('#')]
            line = line.replace('\n','')
            line = line.replace('    ','\t')
            if line == '':
                continue
            preproc_lines.append(line)
        preprocessed_script = '\n'.join(preproc_lines)
    return preprocessed_script

def make_df(df, n_fold, tokenizer):
    train = df[df['fold']!=n_fold].reset_index(drop=True)
    valid = df[df['fold']==n_fold].reset_index(drop=True)

    for phase, df_phase in zip(['train', 'valid'], [train, valid]):
        print(f'------------------------------------- make {phase} set -------------------------------------')
        codes = df_phase['code'].to_list()
        problems = df_phase['problem_num'].unique().tolist()
        problems.sort()

        tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
        bm25 = BM25Okapi(tokenized_corpus)

        total_positive_pairs = []
        total_negative_pairs = []

        for problem in tqdm(problems):
            solution_codes = df_phase[df_phase['problem_num'] == problem]['code']
            positive_pairs = list(combinations(solution_codes.to_list(), 2))

            solution_codes_indices = solution_codes.index.to_list()
            negative_pairs = []
            first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
            negative_code_scores = bm25.get_scores(first_tokenized_code)
            negative_code_ranking = negative_code_scores.argsort()[::-1] # 내림차순
            ranking_idx = 0
         
            for solution_code in solution_codes:
                negative_solutions = []
                while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
                    high_score_idx = negative_code_ranking[ranking_idx]
                    
                    if high_score_idx not in solution_codes_indices:
                        negative_solutions.append(df_phase['code'].iloc[high_score_idx])
                    ranking_idx += 1

                for negative_solution in negative_solutions:
                    negative_pairs.append((solution_code, negative_solution))
 
            total_positive_pairs.extend(positive_pairs)
            total_negative_pairs.extend(negative_pairs)

        pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
        pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

        neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
        neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

        pos_label = [1]*len(pos_code1)
        neg_label = [0]*len(neg_code1)

        pos_code1.extend(neg_code1); total_code1 = pos_code1
        pos_code2.extend(neg_code2); total_code2 = pos_code2
        pos_label.extend(neg_label); total_label = pos_label

        pair_data = pd.DataFrame(data={
            'code1':total_code1,
            'code2':total_code2,
            'similar':total_label
        })
        pair_data = pair_data.sample(frac=1).reset_index(drop=True)

        pair_data.to_csv(f'./data/df_{phase}_{n_fold}fold.csv',index=False)

#%%
##### .py -> csv
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    
    code2df = []
    train_folder = glob('./data/code/*')
    for fold in train_folder:
        pro_num = Path(fold).stem
        problem = glob(f'{fold}/*')
        for script in problem:
            preprocessed_script = preprocess(script)
            code2df.append((preprocessed_script, pro_num))
    code2df = pd.DataFrame(code2df, columns=['code', 'problem_num'])
    code2df.to_csv('./data/code2df.csv', index=False)

    # code2df = pd.read_csv('./data/code2df.csv')

    code2df['len'] = code2df['code'].apply(lambda x: len(tokenizer.tokenize(x)))
    df_train = code2df[code2df['len'] <= 512].reset_index(drop=True)
    
    ratio=0.15
    df_train = df_train.groupby('problem_num').apply(sampling_func, sample_pct=ratio).reset_index(drop=True)
    
    kf = StratifiedKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train, y=df_train['problem_num'])):
        df_train.loc[val_idx, 'fold'] = fold
    
    make_df(df_train, 0, tokenizer)