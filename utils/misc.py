from easydict import EasyDict
import yaml
import os
import numpy as np
import random
import torch


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, save_path, val_fold, patience=10, verbose=False, delta=0):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.local_optimum_pearson_train  = None
        self.local_optimum_spearman_train = None
        self.local_optimum_pearson_val  = None
        self.local_optimum_spearman_val = None
        self.val_fold = val_fold

    def __call__(self, val_loss, model, pearson_train, spearman_train, pearson_val, spearman_val):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.local_optimum_pearson_train = pearson_train
            self.local_optimum_spearman_train = spearman_train
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.early_stop = False
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.local_optimum_pearson_train = pearson_train
            self.local_optimum_spearman_train = spearman_train
            self.local_optimum_pearson_val = pearson_val
            self.local_optimum_spearman_val = spearman_val

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, '{}_fold.pth'.format(self.val_fold))
        torch.save(model.state_dict(), path)	

def consume_random_state(steps=2024):
    for _ in range(steps):
        np.random.rand()
        torch.rand(1)
        torch.tensor(1.0, device='cuda').normal_()
