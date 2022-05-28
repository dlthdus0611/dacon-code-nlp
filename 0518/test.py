#%%
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = "microsoft/graphcodebert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
state_dict = torch.load('./results/runs3/checkpoint-8904/pytorch_model.bin')

model = RobertaForSequenceClassification.from_pretrained(
        MODEL, 
        num_labels = 2,   
        output_attentions = False, 
        output_hidden_states = False
        ).to(device)
model.load_state_dict(state_dict, strict=False)
#%%
test_trainer = Trainer(model)
testset = load_dataset(
    "csv",
    data_files={
        "test": "../data/test.csv",
    },
)
testset = testset['test'].map(lambda x: tokenizer(x["code1"], x['code2'], max_length=512, padding='max_length', truncation=True), remove_columns=['pair_id', 'code1', 'code2'], batched=True)

#%%
raw_pred, _, _ = test_trainer.predict(testset)
y_pred = np.argmax(raw_pred, axis=1)

sub = pd.read_csv('../data/sample_submission.csv')
sub['similar'] = y_pred
# %%
sub.to_csv('submission2_non_sim.csv',index=False)

