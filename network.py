import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

#inspiration: https://arxiv.org/pdf/1803.01271.pdf and accomanying source code https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, pad, dropout = 0.2):
        super(ResidualBlock1D, self).__init__()
        pad0 = torch.nn.ZeroPad2d((pad, 0))
        conv0 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation))
        relu0 = nn.ReLU(inplace=True)
        dropout0 = nn.Dropout(dropout)

        pad1 = torch.nn.ZeroPad2d((pad, 0))
        conv1 = nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, dilation=dilation))
        relu1 = nn.ReLU(inplace=True)
        dropout1 = nn.Dropout(dropout)

        self.layers = nn.Sequential(pad0, conv0, relu0, dropout0, pad1, conv1, relu1, dropout1)

        self.channel_scaling = None
        if in_channels != out_channels:
            self.channel_scaling = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1))


    def forward(self, x):
        res = x
        out = self.layers(x)
        if self.channel_scaling != None:
            res = self.channel_scaling(x)
        return res + out
        

class TimeSeriesCNN(nn.Module):
    def __init__(self, n_layers, input_size, n_channels):
        super(TimeSeriesCNN, self).__init__()
        blocks = []
        for i in range(n_layers):
            dil = i+1
            n_prev_channels = n_channels if i != 0 else input_size
            blocks += [ResidualBlock1D(n_prev_channels, n_channels, 3, dil, pad=dil*2, dropout=0.2)]
        self.residual_blocks = nn.Sequential(*blocks)


    def forward(self, x):
        out = self.residual_blocks(x)
        return out


if __name__ == "__main__":
    import numpy as np
    from torch.autograd import Variable
    x = np.arange(1*2*3).reshape(1,2,3)
    tensor = Variable(torch.zeros(1,5,9))
    print(tensor)
    block = ResidualBlock1D(5, 5, 3, 1, 2)
    print(block(tensor))

    model = TimeSeriesCNN(n_layers = 12, input_size = 5, n_channels = 128)
    print(model(tensor))