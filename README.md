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

## Dataset Preparation
For training, we need to prepare a preprocessed dataset in `.csv` format. First, we converted the `.py` files into `.csv` format. Afterward, we performed preprocessing on the raw code data and split it into the training, development, and test sets.

```python
python ./data/prepro.py
```

## Training
```python
python main.py

Arguments:
  --ver : SimCSE version for training (default:unsup)
  --initial_lr : initial learning rate (default:3e-5)
  --epoch : the number of epoch (default:10)
  --batch_size : the number of batch (default:8)
  --exp_num : experiment number (default:1)
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
