import os
import torch
import pprint
import random
import warnings
from utils import *
import numpy as np
from config import getConfig
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

warnings.filterwarnings('ignore')
args = getConfig()

def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    save_path = os.path.join(args.model_path, (args.exp_num).zfill(3))

    # if args.logging:
    #     api = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4OTRmMzMyMC0wNDA0LTRlZDAtYTg1Ni0zZTU3NDg3NGQ3YTYifQ=="
    #     neptune.init("dlthdus8450/anam", api_token=api)
    #     temp = neptune.create_experiment(name=args.experiment, params=vars(args))
    #     experiment_num = str(temp).split('-')[-1][:-1] 
    #     neptune.append_tag(args.tag)
    #     save_path = os.path.join(args.model_path, experiment_num.zfill(3))
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained = args.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    
    state_dict = torch.load(args.simcse_model)
    del state_dict['mlp.dense.weight']
    del state_dict['mlp.dense.bias']

    model = RobertaForSequenceClassification.from_pretrained(
            pretrained, 
            num_labels = 2,   
            output_attentions = False, 
            output_hidden_states = False
            ).to(device)

    model.load_state_dict(state_dict, strict=False)

    dataset = make_dataset('df_train_0fold', 'df_valid_0fold', tokenizer)
    _collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    training_args = TrainingArguments(
        save_path,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        do_train=True,
        do_eval=True,
        save_total_limit = 2,
        logging_dir = os.path.join(args.model_path, args.exp_num, 'logs'),
        save_strategy = "epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True
    )
    # Create model directory
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )
    
    trainer.train()
    
if __name__ == '__main__':
    main(args)
