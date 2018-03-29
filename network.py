import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

#inspiration: https://arxiv.org/pdf/1803.01271.pdf and accomanying source code https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    
class ResidualBottleneck1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, pad, dropout = 0.2):
        super(ResidualBottleneck1D, self).__init__()
        conv0 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels//4, kernel_size=1, stride=1))
        relu0 = nn.ReLU(inplace=True)
        dropout0 = nn.Dropout(dropout)

        pad = torch.nn.ZeroPad2d((pad, 0))
        conv1 = nn.utils.weight_norm(nn.Conv1d(out_channels//4, out_channels//4, kernel_size, stride=1, dilation=dilation))
        relu1 = nn.ReLU(inplace=True)
        dropout1 = nn.Dropout(dropout)

        conv2 = nn.utils.weight_norm(nn.Conv1d(out_channels//4, out_channels, kernel_size=1, stride=1))
        relu2 = nn.ReLU(inplace=True)
        dropout2 = nn.Dropout(dropout)

        self.layers = nn.Sequential(conv0, relu0, dropout0, pad, conv1, relu1, dropout1, conv2, relu2, dropout2)

        self.channel_scaling = None
        if in_channels != out_channels:
            self.channel_scaling = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1))


    def forward(self, x):
        res = x
        out = self.layers(x)
        if self.channel_scaling != None:
            res = self.channel_scaling(x)
        return res + out
        

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


class RadicalResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, dropout_chance = 0.2):
        super(RadicalResidualBlock, self).__init__()
        depth = 8
        dilations = [int(2**x) for x in range(depth)]
        self.layers = nn.ModuleList()
        for i, dil in enumerate(dilations):
            pad = torch.nn.ZeroPad2d((2*dil, 0))
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != depth-1 else output_size
            conv = nn.utils.weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=1, dilation=dil))
            relu = nn.ReLU()
            dropout = nn.Dropout(dropout_chance)
            self.layers.append(nn.Sequential(pad, conv, relu, dropout))

        self.scaling_in = None
        if input_size != hidden_size:
            self.scaling_in = nn.utils.weight_norm(nn.Conv1d(input_size, hidden_size, kernel_size=1, stride=1))
        self.scaling_out = None
        if output_size != hidden_size:
            self.scaling_out = nn.utils.weight_norm(nn.Conv1d(hidden_size, output_size, kernel_size=1, stride=1))
        self.scaling_large = None
        if output_size != input_size:
            self.scaling_large = nn.utils.weight_norm(nn.Conv1d(input_size, output_size, kernel_size=1, stride=1))


    def forward(self, x):
        res0 = x
        res1 = self.layers[0](res0)
        res2 = self.layers[1](res1)
        if self.scaling_in != None:
            res2 += self.scaling_in(res0)
        else:
            res2 += res0
        res3 = self.layers[2](res2)
        res3 += res1
        res4 = self.layers[3](res3)
        res4 += res2
        res5 = self.layers[4](res4)
        res5 += res3
        res6 = self.layers[5](res5)
        res6 += res4
        res7 = self.layers[6](res6)
        res7 += res5
        res8 = self.layers[7](res7)
        if self.scaling_out != None:
            res8 += self.scaling_out(res6)
        else:
            res8 += res6
        if self.scaling_large != None:
            res8 += self.scaling_large(res0)
        else:
            res8 += res0
        return res8


class DoubleResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, dilation, dropout_chance = 0.2):
        super(DoubleResidualBlock, self).__init__()
        depth = 2
        self.layers = nn.ModuleList()
        for i in range(depth):
            pad = torch.nn.ZeroPad2d((2*dilation, 0))
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != depth-1 else output_size
            conv = nn.utils.weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=1, dilation=dilation))
            relu = nn.ReLU()
            dropout = nn.Dropout(dropout_chance)
            self.layers.append(nn.Sequential(pad, conv, relu, dropout))

        self.scaling = None
        if input_size != output_size:
            self.scaling = nn.utils.weight_norm(nn.Conv1d(input_size, output_size, kernel_size=1, stride=1))


    def forward(self, middle_x, last_x):
        res0 = last_x
        res1 = self.layers[0](last_x)
        if middle_x is not None:
            res1 += middle_x
        res2 = self.layers[1](res1)
        if self.scaling != None:
            res2 += self.scaling(res0)
        else:
            res2 += res0
        return res1, res2
    
    
class TimeSeriesCNN(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, output_size):
        super(TimeSeriesCNN, self).__init__()
        blocks = []
        for i in range(n_layers):
            dil = 1 + 3*i
            #dil = 2**i
            n_prev_channels = hidden_size if i != 0 else input_size
            n_channels = hidden_size if i != n_layers-1 else output_size
            blocks += [ResidualBlock1D(n_prev_channels, n_channels, 3, dil, pad=dil*2, dropout=0.2)]
        self.residual_blocks = nn.Sequential(*blocks)


    def forward(self, x):
        out = self.residual_blocks(x)
        return out


class RadicalTimeSeriesCNN(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, output_size):
        super(RadicalTimeSeriesCNN, self).__init__()
        blocks = []
        for i in range(n_layers):
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != n_layers-1 else output_size
            blocks += [RadicalResidualBlock(c_in, hidden_size, c_out, 3, dropout_chance = 0.2)]

        self.residual_blocks = nn.Sequential(*blocks)


    def forward(self, x):
        out = self.residual_blocks(x)
        return out


class DoubleTimeSeriesCNN(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, output_size):
        super(DoubleTimeSeriesCNN, self).__init__()
        self.n_layers = n_layers
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dil = 1 + 3*i
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != n_layers-1 else output_size
            self.blocks.append(DoubleResidualBlock(c_in, hidden_size, c_out, 3, dil,  dropout_chance = 0.2))


    def forward(self, x):
        middle_x = None
        last_x = x
        for i in range(self.n_layers):
            middle_x, last_x = self.blocks[i](middle_x, last_x)
        return last_x


if __name__ == "__main__":
    import numpy as np
    from torch.autograd import Variable
    x = np.arange(1*2*3).reshape(1,2,3)
    tensor = Variable(torch.zeros(100,1,28*28))
    print(tensor)
    model = DoubleTimeSeriesCNN(n_layers=7, input_size=1, hidden_size=64, output_size=10)
    print(model)
