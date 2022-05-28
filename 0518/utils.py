import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def encode(tokenizer, data):
    outputs = tokenizer(data["code1"], data['code2'], max_length=512, padding='max_length', truncation=True)
    if 'similar' in data:
        outputs["labels"] = data['similar']
    return outputs

def make_dataset(trainset, validset, tokenizer):
    dataset = load_dataset(
        "csv",
        data_files={
            "train": f"../data/{trainset}.csv",
            "val": f"../data/{validset}.csv",
        },
    )
    dataset = dataset.map(lambda x: encode(tokenizer, x), remove_columns=['code1', 'code2', 'similar'], batched=True)

    dataset.set_format(
        type="torch",
        columns=['input_ids', 'attention_mask', 'labels'],
        device=device,
    )
    return dataset

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}