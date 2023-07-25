#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np

class Collator(object):
    def __init__(self, args, phase:str):
        super(Collator, self).__init__()

        self.args  = args
        self.phase  = phase
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def collate(self, batch):
        
        lengths_qa = [len(sample['input_ids']) for sample in batch]
        batch_input_ids = []
        batch_attention_mask = []
        batch_segment_ids = []

        for idx, sample in enumerate(batch):
            padding_qa  = [0] * (max(lengths_qa) - len(sample['input_ids']))

            sample['input_ids']  += padding_qa
            sample['attention_mask'] += padding_qa
            sample['token_type_ids'] += padding_qa

            batch_input_ids.append(sample['input_ids'])
            batch_attention_mask.append(sample['attention_mask'])
            batch_segment_ids.append(sample['token_type_ids'])

        batch_input_ids = torch.LongTensor(batch_input_ids).to(self.device)
        batch_attention_mask = torch.LongTensor(batch_attention_mask).to(self.device)
        batch_segment_ids = torch.LongTensor(batch_segment_ids).to(self.device)

        return batch_input_ids, batch_attention_mask, batch_segment_ids

    def __call__(self, batch):

        if self.phase != 'train':
            
            re1 = self.collate([i['input1']  for i in batch])
            re2 = self.collate([i['input2']  for i in batch])
            batch_label = torch.LongTensor([i['label']  for i in batch]).to(self.device)
            
            results = {
                        'results1': re1,
                        'results2': re2,
                        'labels': batch_label
                    }
        else:

            if self.args.ver == 'unsup':

                results = self.collate(batch)

            elif self.args.ver == 'sup':
                
                re1 = self.collate([i['sample']  for i in batch])
                re2 = self.collate([i['positive']  for i in batch])
                re3 = self.collate([i['negative']  for i in batch])
                
                results = {
                            'results1': re1,
                            'results2': re2,
                            'results3': re3
                        }
            
        return results