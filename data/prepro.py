import random
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from rank_bm25 import BM25Okapi
from itertools import combinations
from transformers import RobertaTokenizer
from sklearn.model_selection import StratifiedKFold

from ..utils import fix_seed

def preprocess(script):

    with open(script,'r',encoding='utf-8') as file:
        lines = file.readlines()
        preproc_lines = []

        for line in lines:
            if line.lstrip().startswith('#'):
                continue
            elif (line.lstrip().startswith('from')) or (line.lstrip().startswith('import')):
                if line.__contains__(';'):
                    line = line[line.index(';')+1:]
                else:
                    continue
            elif line.lstrip().startswith('if __name__'):
                continue
            
            line = line.strip()
            if '#' in line:
                line = line[:line.index('#')]
            line = line.replace('\n','')
            line = line.replace('    ','\t')
            if line == '':
                continue
            preproc_lines.append(line)
        preprocessed_script = '\n'.join(preproc_lines)

    return preprocessed_script

def convert_py2csv(path):

    code2df = []
    train_folder = glob(f'{path}')

    for fold in train_folder:
        pro_num = Path(fold).stem
        problem = glob(f'{fold}/*')
        for script in problem:
            preprocessed_script = preprocess(script)
            code2df.append((preprocessed_script, pro_num))

    code2df = pd.DataFrame(code2df, columns=['code', 'problem_num'])
    code2df.to_csv('./data/code2df.csv', index=False)

    return code2df

def split_data(df):

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(df, df['problem_num'])):
        df.loc[test_idx, 'fold'] = fold

    df_train = df.loc[(df['fold'] != 0) & (df['fold'] != 1) & (df['fold'] != 2)].drop(columns=['fold'])
    df_dev   = df.loc[df['fold'] == 0].drop(columns=['fold'])
    df_test  = df.loc[(df['fold'] == 1) | (df['fold'] == 2)].drop(columns=['fold'])

    df_train.to_csv('./train.csv', index=False)
    df_dev.to_csv('./dev.csv', index=False)
    df_test.to_csv('./test.csv', index=False)

    return df_train, df_dev, df_test

def make_pair_df(df, tokenizer):

    codes = df['code'].to_list()
    problems = df['problem_num'].unique().tolist()
    problems.sort()

    tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
    bm25 = BM25Okapi(tokenized_corpus)

    total_positive_pairs = []
    total_negative_pairs = []

    for problem in tqdm(problems):
        solution_codes = df[df['problem_num'] == problem]['code']
        positive_pairs = list(combinations(solution_codes.to_list(),2))

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
                    negative_solutions.append(df['code'].iloc[high_score_idx])
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

    pos_code1.extend(neg_code1)
    pos_code2.extend(neg_code2)
    pos_label.extend(neg_label)
    
    total_code1 = pos_code1
    total_code2 = pos_code2
    total_label = pos_label

    pair_data = pd.DataFrame(data={
        'code1':total_code1,
        'code2':total_code2,
        'similar':total_label
    })
    pair_data = pair_data.sample(frac=1)

    return pair_data

def make_sup_df(df):

    codes = df['code'].to_list()

    total_positive = []
    total_negative = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        problem = row['problem_num']

        positive_codes = df[df['problem_num'] == problem]['code'].values
        negative_codes = df[df['problem_num'] != problem]['code'].values
        positive = random.choice(positive_codes)
        negative = random.choice(negative_codes)

        total_positive.append(positive)
        total_negative.append(negative)

    pair_data = pd.DataFrame(data={
        'code':codes,
        'positive':total_positive,
        'negative':total_negative
    })
    
    return pair_data

if __name__ == '__main__':
    
    fix_seed(seed=42)
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    code2df = convert_py2csv('./code/*')    
    df_train, df_dev, df_test = split_data(code2df)

    df_train_sup = make_sup_df(df_train)
    df_dev_pair  = make_pair_df(df_dev, tokenizer)
    df_test_pair = make_pair_df(df_test, tokenizer)

    df_train_sup.to_csv('train_sup.csv', index=False)
    df_dev_pair.to_csv('dev_pair.csv', index=False)
    df_test_pair.to_csv('test_pair.csv', index=False)