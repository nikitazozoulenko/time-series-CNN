import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

#inspiration: https://arxiv.org/pdf/1803.01271.pdf

class ResidualBlock1D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, dilation, dropout_chance = 0.2):
        super(ResidualBlock1D, self).__init__()
        depth = 2
        self.layers = nn.ModuleList()
        for i in range(depth):
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != depth-1 else output_size
            pad = torch.nn.ZeroPad2d(((kernel_size-1)*dilation, 0))
            conv = nn.utils.weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=1, dilation=dilation))
            relu = nn.ReLU()
            dropout = nn.Dropout(dropout_chance)
            self.layers.append(nn.Sequential(pad, conv, relu, dropout))

        self.channel_scaling = None
        if input_size != output_size:
            self.channel_scaling = nn.utils.weight_norm(nn.Conv1d(input_size, output_size, kernel_size=1, stride=1))


    def forward(self, x):
        res = x
        x = self.layers[0](x)
        x = self.layers[1](x)
        if self.channel_scaling != None:
            res = self.channel_scaling(x)
        return res + x


class TCNSingle(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, output_size, kernel_size, dilation_lambda = lambda l: 1+l):
        super(TCNSingle, self).__init__()
        blocks = []
        for i in range(n_layers):
            dil = dilation_lambda(i)
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != n_layers-1 else output_size
            blocks += [ResidualBlock1D(c_in, hidden_size, c_out, kernel_size, dil, dropout_chance = 0.2)]
        self.residual_blocks = nn.Sequential(*blocks)


    def forward(self, x):
        out = self.residual_blocks(x)
        return out


class RadicalResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, dilation, dropout_chance = 0.2):
        super(RadicalResidualBlock, self).__init__()
        depth = 4
        self.layers = nn.ModuleList()
        for i in range(depth):
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != depth-1 else output_size
            pad = torch.nn.ZeroPad2d(((kernel_size-1)*dilation, 0))
            conv = nn.utils.weight_norm(nn.Conv1d(c_in, c_out, kernel_size, stride=1, dilation=dilation))
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
        if self.scaling_out != None:
            res4 += self.scaling_out(res2)
        else:
            res4 += res2
        if self.scaling_large != None:
            res4 += self.scaling_large(res0)
        else:
            res4 += res0
        return res4


class TCNRadical(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, output_size, kernel_size, dilation_lambda = lambda l: 1+l):
        super(TCNRadical, self).__init__()
        blocks = []
        for i in range(n_layers):
            dil = dilation_lambda(i)
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != n_layers-1 else output_size
            blocks += [RadicalResidualBlock(c_in, hidden_size, c_out, kernel_size, dil, dropout_chance = 0.2)]

        self.residual_blocks = nn.Sequential(*blocks)


    def forward(self, x):
        out = self.residual_blocks(x)
        return out


class DoubleResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, dilation, dropout_chance = 0.2):
        super(DoubleResidualBlock, self).__init__()
        depth = 2
        self.layers = nn.ModuleList()
        for i in range(depth):
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != depth-1 else output_size
            pad = torch.nn.ZeroPad2d(((kernel_size-1)*dilation, 0))
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


class TCNDouble(nn.Module):
    def __init__(self, n_layers, input_size, hidden_size, output_size, kernel_size, dilation_lambda = lambda l: 1+l):
        super(TCNDouble, self).__init__()
        self.n_layers = n_layers
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dil = dilation_lambda(i)
            c_in = hidden_size if i != 0 else input_size
            c_out = hidden_size if i != n_layers-1 else output_size
            self.blocks.append(DoubleResidualBlock(c_in, hidden_size, c_out, kernel_size, dil, dropout_chance = 0.2))


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
    model = TCNRadical(n_layers=6, input_size=1, hidden_size=32, output_size=10, kernel_size=3, 
                        dilation_lambda = lambda l: 1+l*3)
    print(model(tensor))