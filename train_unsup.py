import os
import time
import utils
import torch
import random
import neptune
import logging
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch_optimizer as optim
from sklearn.metrics import f1_score
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from data_modules import *

class Trainer:

    def __init__(self, args, info):
        super(Trainer, self).__init__()

        self.args = args
        self.run = info.run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Logging
        self.save_path = info.save_path
        log_file = os.path.join(info.save_path, 'log.log')
        self.logger = utils.get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)
        self.logger.info(args.tag)

        # Model
        self.model = info.model.to(self.device)

        # DataLoader
        df_train = pd.read_csv('./data/train_sup.csv')
        df_dev   = pd.read_csv('./data/dev_pair.csv')
        df_test  = pd.read_csv('./data/test_pair.csv')

        self.train_loader = get_loader(args, df_train, info.tokenizer, phase='train')
        self.dev_loader   = get_loader(args, df_dev, info.tokenizer, phase='dev')
        self.test_loader  = get_loader(args, df_test, info.tokenizer, phase='test')

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer & Scheduler
        self.optimizer = optim.Lamb(self.model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
        
        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = utils.WarmUpLR(self.optimizer, iter_per_epoch * args.warm_epoch)

        if args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestone, gamma=args.lr_factor, verbose=False)
        elif args.scheduler == 'cos':
            tmax = args.tmax # half-cycle 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = tmax, eta_min=args.min_lr, verbose=False)
    
    def train(self):

        # Train / Validate
        best_acc = 0
        start = time.time()

        for epoch in range(1, self.args.epochs+1):
            self.logger.info(f'Epoch:[{epoch:03d}/{self.args.epochs:03d}]')
            self.epoch = epoch

            # Training
            train_loss = self.train_one_epoch()
            val_loss, val_acc, val_f1 = self.validate()

            self.logger.info(f'\nTrain Loss:{train_loss:.4f}')
            self.logger.info(f'\nVal Loss  :{val_loss:.4f} | Val Acc:{val_acc:.4f} | Val F1:{val_f1:.4f}')
            
            # Save models
            if val_acc > best_acc :
                best_acc   = val_acc
                best_epoch = epoch
            
                # Model weight in Multi_GPU or Single GPU
                state_dict = self.model.module.state_dict() if self.args.multi_gpu else self.model.state_dict()
                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                    }, os.path.join(self.save_path, f'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------') 
                patience = 0  
            else:
                patience +=1

            if patience == self.args.patience:
                break

        self.logger.info(f'\nBest Val Epoch:{best_epoch} | Val Acc:{best_acc:.4f}')

        test_acc, test_f1 = self.test()
        self.logger.info(f'\nTest Acc:{test_acc:.4f} | Test F1:{test_f1:.4f}')

        end = time.time()
        self.logger.info(f'Total Process time:{(end - start) / 60:.3f}Minute')

    def train_one_epoch(self):
        
        self.model.train()
        train_loss = utils.AvgMeter()

        for batch in tqdm(self.train_loader):

            labels = self.model.create_label(batch)
            cos_sim = self.model(batch) 
            loss = self.criterion(cos_sim, labels)
            loss.backward()

            # Gradient Clipping
            if self.args.clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clipping)

            if self.epoch > self.args.warm_epoch:
                self.scheduler.step()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            # log
            train_loss.update(loss.mean().item(), n=labels.size(0))

        return train_loss.avg

    def validate(self):
        acc = 0
        preds_list, targets_list = [], []
        self.model.eval()

        sigmoid = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            val_loss = utils.AvgMeter()

            for batch in tqdm(self.dev_loader):
                labels = batch['labels'].float()
                cos_sim = self.model(batch, phase='dev')
                loss = criterion(cos_sim, labels)

                # log
                val_loss.update(loss.mean().item(), n=labels.size(0))
                
                preds = (sigmoid(cos_sim) > 0.5)
                preds_list.extend(preds.cpu().detach().numpy())
                targets_list.extend(labels.cpu().detach().numpy())

                acc += (preds.cpu().detach() == labels.cpu().detach()).sum().item()

            acc /= len(self.dev_loader.dataset)
            f1  = f1_score(np.array(targets_list), np.array(preds_list), average='macro')

        return val_loss.avg, acc, f1
    
    def test(self):
        acc = 0
        preds_list, targets_list = [], []
        self.model.eval()

        sigmoid = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():

            for batch in tqdm(self.test_loader):
                labels = batch['labels'].float()
                cos_sim = self.model(batch, phase='dev')
                loss = criterion(cos_sim, labels)

                # log
                preds = (sigmoid(cos_sim) > 0.5)
                preds_list.extend(preds.cpu().detach().numpy())
                targets_list.extend(labels.cpu().detach().numpy())

                acc += (preds.cpu().detach() == labels.cpu().detach()).sum().item()

            acc /= len(self.test_loader.dataset)
            f1  = f1_score(np.array(targets_list), np.array(preds_list), average='macro')

        return acc, f1