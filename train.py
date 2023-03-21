import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImgClassifier
from trainer import Trainer

from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes

def define_argparse():
    #인스턴스 생성
    p = argparse.ArgumentParser()
    
    p.add.argment('--model_fn', required=True)
    p.add.argment('--gpu_id', type=int, default=0 if torch.cuda.is_avaliable() else -1) #cpu가 있으면 실행, 없으면
    p.add.argment('--train.ratio', type=float, default=.8)
    p.add.argment('--batch_size', default=256)
    p.add.argment('--n_epochs', default=20)
    p.add.argment('--n_layers', default=5)
    p.add.argment('--use_dropout', action='store_true')
    p.add.argment('--dropout_p', type=float, default=.3)
    p.add.argment('--verbose', type=int, default=1)
    
    config = p.parse_args()
    
    return config

def main(config):
    # 입력된 gpu를 사용, 없으면 cpu
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda : %d' % config.gpu_id)
    