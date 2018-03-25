import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from network import TimeSeriesCNN
from data_feeder import DataFeeder
from logger import Logger

        
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


def evaluate_acc(batch_size, model, mnist_test, i, val_loss, val_acc):
    model.eval()

    images, labels = mnist_test
    accs = []
    losses = []
    for j in range(len(images)//batch_size):
        batch_images = images[j*batch_size:(j+1)*batch_size]
        batch_labels = labels[j*batch_size:(j+1)*batch_size]
        out = model(batch_images)[:,:,-1]
        _, index = out.topk(1, dim=1)
        accs += [torch.mean((index.squeeze() == batch_labels).float())]
        losses += [F.cross_entropy(out, batch_labels)]

    acc = torch.mean(torch.stack(accs))
    loss = torch.mean(torch.stack(losses))
    val_loss.write_log(loss, i)
    val_acc.write_log(acc, i)

    model.train()
    
        
def train(model, optimizer, images, labels, i, train_logger):
    optimizer.zero_grad()
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
    mnist_train = datasets.MNIST("/hdd/Data/MNIST/", train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = datasets.MNIST("/hdd/Data/MNIST/", train=False, transform=transforms.ToTensor(), download=True)
    data_feeder = DataFeeder(mnist_train, preprocess_workers = 1, cuda_workers = 1, cpu_size = 10,
                 cuda_size = 10, batch_size = batch_size, use_cuda = True, volatile = False)
    data_feeder.start_queue_threads()
    test_data = make_batch(len(mnist_test), 0, mnist_test, use_cuda = True, volatile = True)

    model = TimeSeriesCNN(n_layers=18, input_size=1, n_channels=64, output_size=10).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00001)

    train_loss = Logger("train_losses.txt")
    val_loss = Logger("val_losses.txt")
    val_acc = Logger("val_acc.txt")

    for i in range(80001):
        images, labels = data_feeder.get_batch()
        train(model, optimizer, images, labels, i, train_loss)
        if i % 100 == 0:
            print(i)
            evaluate_acc(batch_size, model, test_data, i, val_loss, val_acc)
        if i in [40000, 60000]:
            decrease_lr(optimizer)
        if i % 5000 == 0:
            torch.save(model.state_dict, "savedir/cnn_it"+str(i//1000)+"k.pth")
            
    data_feeder.kill_queue_threads()

    
if __name__ == "__main__":
    main()
