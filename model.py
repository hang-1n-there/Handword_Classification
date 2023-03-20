import torch
import torch.nn as nn

# Block의 순서 (Layer -> activation function -> batch_norm and dropout)
class Block(nn.Module):
    def __init__(self, input_size, output_size, use_batch_norm=True, dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        # batch nomalization, dropout 중 하나를 고르는 함수
        def get_regularizer(use_batch_norm, size):
            return nn.BatchNormld(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size),
        )
    def forward(self, x):
        # |x| = (batch_size * input_size)
        y = self.block(x)
        # |y| = (batch_size * output_size)
        
        return y

class ImgClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[500,400,300,200,100], use_batch_norm=True, dropout_p=.3):
        
        super().__init__()
        # assert 조건, 메세지 | assert의 조건이 true가 아니면 error와 메세지를 반환
        assert len(hidden_sizes) > 0, 'You need specify hidden layers'
        
        last_hidden_size = input_size
        blocks=[]
        
        for hidden_size in hidden_sizes:
            # last_hidden_size = Q(t), hidden_size = Q(t+1)
            blocks+=[Block(last_hidden_size, hidden_size, use_batch_norm, dropout_p)]
            last_hidden_size=hidden_size
        
        self.layers = nn.Sequential(
            blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )
    def forward(self, x):
        # |x| = (batch_size * input_size)
        y = self.layers(x)
        # |y| = (batch_size * output_size)
        return y
