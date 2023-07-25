import os
import pprint
import neptune
import warnings
from easydict import EasyDict
from transformers import RobertaTokenizer

from utils import *
from models import *
from config import getConfig
# from train_unsup import Trainer

warnings.filterwarnings('ignore')
args = getConfig()

def main(args):

    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    fix_seed(args.seed)

    if args.logging:
        api = "api_token"
        run = neptune.init_run(project="ID/Project", api_token=api, name=args.experiment, tags=args.tag.split(','))
        run.assign({'parameters':vars(args)})
        exp_num = run._sys_id.split('-')[-1].zfill(3)
    else:
        run = None
        exp_num = args.exp_num
    
    # Create model directory
    save_path = os.path.join(args.model_path, exp_num)
    os.makedirs(save_path, exist_ok=True)
    
    config    = RobertaConfig.from_pretrained('microsoft/codebert-base')
    model     = RobertaModelForSimCSE.from_pretrained('microsoft/codebert-base', config=config, temperature=0.05)
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    info = EasyDict({
                     'run': run,
                     'model': model,
                     'save_path': save_path,
                     'tokenizer': tokenizer
                    })
    if args.ver == 'sup':
        from train_sup import Trainer
    else:
        from train_unsup import Trainer
        
    Trainer(args, info).train()

if __name__ == '__main__':
    main(args)