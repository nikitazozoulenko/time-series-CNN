import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

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


def evaluate_acc(batch_size, model, data, i, val_loss, val_acc, permute):
    model.eval()

    images, labels = data
    accs = []
    losses = []
    for j in range(0, len(images), batch_size):
        #loss
        batch_images = images[j:j+batch_size]
        batch_labels = labels[j:j+batch_size]

        if permute is not None:
            batch_images = batch_images[:,:,permute]
        
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


def main(permute):
    batch_size = 100
    train_data, val_data, test_data = create_train_val_test_split(batch_size)
    val_data = make_batch(len(val_data), 0, val_data, use_cuda = True, volatile = True)
    test_data = make_batch(len(test_data), 0, test_data, use_cuda = True, volatile = True)

    single = TimeSeriesCNN(n_layers=18, input_size=1, hidden_size=64, output_size=10).cuda()
    double = DoubleTimeSeriesCNN(n_layers = 18, input_size=1, hidden_size=64, output_size=10).cuda()
    radical = RadicalTimeSeriesCNN(n_layers=2, input_size=1, hidden_size=64, output_size=10).cuda()
    
    single_test_loss = Logger("single_test_losses.txt")
    single_test_acc = Logger("single_test_acc.txt")
    double_test_loss = Logger("double_test_losses.txt")
    double_test_acc = Logger("double_test_acc.txt")
    radical_test_loss = Logger("radical_test_losses.txt")
    radical_test_acc = Logger("radical_test_acc.txt")
    for i in range(0, 100001, 1000):
        print(i)
        single.load_state_dict(torch.load("savedir/single_it"+str(i//1000)+"k.pth"))
        evaluate_acc(batch_size, single, test_data, i, single_test_loss, single_test_acc, permute)
        double.load_state_dict(torch.load("savedir/double_it"+str(i//1000)+"k.pth"))
        evaluate_acc(batch_size, double, test_data, i, double_test_loss, double_test_acc, permute)
        radical.load_state_dict(torch.load("savedir/radical_it"+str(i//1000)+"k.pth"))
        evaluate_acc(batch_size, radical, test_data, i, radical_test_loss, radical_test_acc, permute)

    
if __name__ == "__main__":
    main(None)
