from .dataset import *
from .collator import Collator

from torch.utils.data import DataLoader

def get_loader(args, df, tokenizer, phase:str):

    collate_fn = Collator(args, phase)

    if phase == 'train':

        if args.ver =='unsup':
            dataset = CodeDataset_unsup(args, df, tokenizer)
        elif args.ver =='sup':
            dataset = CodeDataset_sup(args, df, tokenizer)
        
        data_loader = DataLoader(
                                 dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate_fn,
                                 shuffle=True,
                                 drop_last=True
                                )
    else:

        dataset = CodeDataset_eval(args, df, tokenizer)
        data_loader = DataLoader(
                                 dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate_fn,
                                 shuffle=True,
                                 drop_last=False
                                )

    return data_loader