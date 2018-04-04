import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

from network import ImageAnnotator
from loss import Loss
from lang import Lang
from data_feeder import DataFeeder
from logger import Logger


def train(batch_loss, optimizer, images, captions, num_words):
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

def calc_loss(model, loss, images, captions, num_words, i, train_logger):
    #everything but last EOS token
    pred = model(images, captions[:,:,:-1], test_time = False)
    batch_loss = loss(pred, captions, num_words)
    train_logger.write_log(batch_loss, i)
    return batch_loss


def decrease_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("updated learning rate: new lr:", param_group['lr']/10)
        param_group['lr'] = param_group['lr']/10
    

def main():
    coco_path = "/hdd/Data/MSCOCO2017/images"
    annFile = "/hdd/Data/MSCOCO2017/annotations"
    use_cuda = True

    lang = Lang()
    train_data_feeder = DataFeeder(coco_path+"/train2017/",
                                   annFile+"/captions_train2017.json",
                                   lang, 
                                   preprocess_workers = 4, cuda_workers = 1, 
                                   cpu_size = 15, cuda_size = 2, 
                                   batch_size = 2, use_cuda = use_cuda, use_jitter = True, volatile = False)
    val_data_feeder = DataFeeder(coco_path+"/val2017/",
                                   annFile+"/captions_val2017.json",
                                   lang, 
                                   preprocess_workers = 1, cuda_workers = 1, 
                                   cpu_size = 10, cuda_size = 1, 
                                   batch_size = 1, use_cuda = use_cuda, use_jitter = True, volatile = True)

    train_data_feeder.start_queue_threads()
    val_data_feeder.start_queue_threads()

    model = ImageAnnotator(n_layers=18, hidden_size=256, lang=lang).cuda()
    model.load_state_dict(torch.load("savedir/model_01_it700k.pth"))
    version = "02"
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001)
    loss = Loss().cuda()

    train_logger = Logger("train_losses.txt")
    val_logger = Logger("val_losses.txt")

    for i in range(300001):
        images, captions, num_words = train_data_feeder.get_batch()
        batch_loss = calc_loss(model, loss, images, captions, num_words, i, train_logger)
        train(batch_loss, optimizer, images, captions, num_words)
        if i % 10 == 0:
            print(i)
            model.eval()
            images, captions, num_words = val_data_feeder.get_batch()
            calc_loss(model, loss, images, captions, num_words, i, val_logger)
            model.train()
        #if i in [500000]:
        #    decrease_lr(optimizer)
        if i % 100000 == 0:
            torch.save(model.state_dict(), "savedir/model_"+version+"_it"+str(i//1000)+"k.pth")
            
    train_data_feeder.kill_queue_threads()
    val_data_feeder.kill_queue_threads()

    
if __name__ == "__main__":
    main()
