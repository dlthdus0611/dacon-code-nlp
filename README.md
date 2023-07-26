# dacon-code-nlp
This repository is the solution for [DACON Code similarity judgment AI contest.](https://dacon.io/competitions/official/235900/overview/description)

## Overview
- contrastive learning 기법인 SimCSE를 기반으로 코드 유사성 판단을 위한 embedding space를 구축하고자 함.
- Supervised / Unsupervised 기반의 SimSCE를 수행하여 BERT를 학습하고, test set을 기반으로 평가함.

## Requirements
- Ubuntu 18.04, Cuda 11.1
- tqdm
- numpy
- pandas
- scikit-learn==1.0.2
- transformers==4.19.2
- torch==1.8.0 with cuda 11.1
- torch_optimizer


## Make Dataset
```python
python data/prepro.py
```

## Training
### Supervised
```python
sh train.sh

#1 Supervised
python main.py --ver=sup --initial_lr=4e-5 --batch_size=8 --exp_num=sim_sup

#2 Unsupervised
python main.py --ver=unsup --initial_lr=3e-5 --batch_size=8 --exp_num=sim_unsup
```

## References
```
@misc{gao2021simcse,
    title={SimCSE: Simple Contrastive Learning of Sentence Embeddings},
    author={Tianyu Gao and Xingcheng Yao and Danqi Chen},
    year={2021},
    eprint={2104.08821},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```