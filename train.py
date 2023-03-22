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
    p.add.argment('--train_ratio', type=float, default=.8)
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
    
    x, y = load_mnist(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)
    
    print('Train : ', x[0].shape, y[0].shape)
    print('Valid : ', x[1].shape, y[1].shape)
    
    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0]))+1
    
    model = ImgClassifier(
        input_size = input_size,
        output_size = output_size,
        hidden_sizes = get_hidden_sizes(input_size, output_size, config.n_layers),
        use_batch_norm = not config.use_dropout,
        dropout_p=config.dropout_p,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()
    
    if config.verbose:
        print(model)
        print(optimizer)
        print(crit)
    
    trainer = Trainer(model, optimizer, crit)
    
    trainer.train(
        train_data=(x[0],y[0]),
        valid_data=(x[1],y[0]),
        config=config
    )
    
    #가장 좋은 모델의 파라미터들을 기록
    torch.save({
        'model' : trainer.model.state_dict(),
        'opt' : optimizer.state_dict(),
        'config' : config
    }, config.model_fn)
    
if __name__ == '__main__':
    config = define_argparse
    main(config)