import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

from network import TimeSeriesCNN, RadicalTimeSeriesCNN, DoubleTimeSeriesCNN
from data_feeder import DataFeeder
from logger import Logger

import random

def create_train_val_test_split(batch_size):
    mnist_train = datasets.MNIST("/hdd/Data/MNIST/", train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = datasets.MNIST("/hdd/Data/MNIST/", train=False, transform=transforms.ToTensor(), download=True)

    indices = random.sample(range(60000), 5000)
    train_data = []
    val_data = []
    test_data = []
    for i, ex in enumerate(mnist_train):
        if i in indices:
            val_data += [ex]
        else:
            train_data += [ex]
    for ex in mnist_test:
        test_data += [ex]

    return train_data, val_data, test_data

        
def make_batch(batch_size, i, data, use_cuda = True, volatile = True):
    images = []
    labels = []
    for j in range(batch_size):
        image, label = data[(i+j) % len(data)]
        images += [image]
        labels += [label]
    images_tensor = torch.stack(images).resize_(batch_size, 1, 28*28)
    labels_tensor = torch.LongTensor(labels)
    if use_cuda:
        images_tensor = images_tensor.cuda()
        labels_tensor = labels_tensor.cuda()
    return Variable(images_tensor, volatile=volatile), Variable(labels_tensor, volatile=volatile)


def evaluate_acc(batch_size, model, data, i, val_loss, val_acc):
    model.eval()

    images, labels = data
    accs = []
    losses = []
    for j in range(0, len(images), batch_size):
        #loss
        batch_images = images[j:j+batch_size]
        batch_labels = labels[j:j+batch_size]
        pred = model(batch_images)[:,:,-1]
        losses += [F.cross_entropy(pred, batch_labels)]

        #acc
        _, index = pred.topk(1, dim=1)
        accs += [torch.mean((index.squeeze() == batch_labels).float())]
        
    acc = torch.mean(torch.stack(accs))
    print(acc)
    loss = torch.mean(torch.stack(losses))
    val_loss.write_log(loss, i)
    val_acc.write_log(acc, i)

    model.train()
    
        
def train(model, optimizer, images, labels, i, train_logger):
    optimizer.zero_grad()
    # only take the loss of the last time step
    pred = model(images)[:,:,-1]
    loss = F.cross_entropy(pred, labels)
    loss.backward()
    optimizer.step()

    train_logger.write_log(loss, i)


def decrease_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("updated learning rate: new lr:", param_group['lr']/10)
        param_group['lr'] = param_group['lr']/10
    

def main():
    batch_size = 100
    train_data, val_data, test_data = create_train_val_test_split(batch_size)
    data_feeder = DataFeeder(train_data, preprocess_workers = 1, cuda_workers = 1, cpu_size = 10,
                 cuda_size = 10, batch_size = batch_size, use_cuda = True, volatile = False)
    data_feeder.start_queue_threads()
    val_data = make_batch(len(val_data), 0, val_data, use_cuda = True, volatile = True)
    test_data = make_batch(len(test_data), 0, test_data, use_cuda = True, volatile = True)

    single = TimeSeriesCNN(n_layers=18, input_size=1, hidden_size=64, output_size=10).cuda()
    double = DoubleTimeSeriesCNN(n_layers = 18, input_size=1, hidden_size=64, output_size=10).cuda()
    radical = RadicalTimeSeriesCNN(n_layers=2, input_size=1, hidden_size=64, output_size=10).cuda()
    optimizer_single = optim.SGD(single.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00001)
    optimizer_double = optim.SGD(double.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001)
    optimizer_radical = optim.SGD(radical.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00001)

    single_train_loss = Logger("single_train_losses.txt")
    single_val_loss = Logger("single_val_losses.txt")
    single_val_acc = Logger("single_val_acc.txt")
    double_train_loss = Logger("double_train_losses.txt")
    double_val_loss = Logger("double_val_losses.txt")
    double_val_acc = Logger("double_val_acc.txt")
    radical_train_loss = Logger("radical_train_losses.txt")
    radical_val_loss = Logger("radical_val_losses.txt")
    radical_val_acc = Logger("radical_val_acc.txt")

    for i in range(100001):
        #perm = Variable(torch.from_numpy(np.random.permutation(28*28)).long().cuda(), requires_grad=False)
        #print(perm)
        images, labels = data_feeder.get_batch()
        #print(images[:,:,perm])
        train(single, optimizer_single, images, labels, i, single_train_loss)
        train(double, optimizer_double, images, labels, i, double_train_loss)
        train(radical, optimizer_radical, images, labels, i, radical_train_loss)
        if i % 100 == 0:
            print(i)
            evaluate_acc(batch_size, single, val_data, i, single_val_loss, single_val_acc)
            evaluate_acc(batch_size, double, val_data, i, double_val_loss, double_val_acc)
            evaluate_acc(batch_size, radical, val_data, i, radical_val_loss, radical_val_acc)
        if i in [70000, 80000, 90000]:
            decrease_lr(optimizer_single)
            decrease_lr(optimizer_double)
            decrease_lr(optimizer_radical)
        if i % 1000 == 0:
            torch.save(single.state_dict(), "savedir/single_it"+str(i//1000)+"k.pth")
            torch.save(double.state_dict(), "savedir/double_it"+str(i//1000)+"k.pth")
            torch.save(radical.state_dict(), "savedir/radical_it"+str(i//1000)+"k.pth")
            
    data_feeder.kill_queue_threads()

    
if __name__ == "__main__":
    main()
