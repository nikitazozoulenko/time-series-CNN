"""Taken and modified from 
https://www.sagivtech.com/2017/09/19/optimizing-pytorch-training-code/
credits goes to them"""

import numpy as np
from threading import Thread
import os


import time
import threading
import sys
from queue import Empty,Full,Queue

import torch
from torch.autograd import Variable
from torchvision import transforms, datasets

class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """
    def __init__(self):
        self.to_kill = False
	
    def __call__(self):
        return self.to_kill
	
    def set_tokill(self,tokill):
        self.to_kill = tokill
	
def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    while tokill() == False:
        for i, batch in enumerate(dataset_generator):
            #We fill the queue with new fetched batch until we reach the max size.
            batches_queue.put(batch, block=True)
            
            if tokill() == True:
                return

def threaded_cuda_batches(tokill,cuda_batches_queue,batches_queue, use_cuda, volatile):
    """Thread worker for transferring pytorch tensors into
    GPU. batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        batch_images, batch_labels = batches_queue.get(block=True)

        if use_cuda:
            batch_images = batch_images.cuda()
            batch_labels = batch_labels.cuda()

        batch_images = Variable(batch_images, volatile = volatile)
        batch_labels = Variable(batch_labels, volatile = volatile)
        cuda_batches_queue.put((batch_images, batch_labels), block=True)
        
        if tokill() == True:
            return


class BatchMaker(object):
    def __init__(self, pdata, batch_size):
        self.pdata = pdata
        self.indices = np.arange(len(pdata))
        np.random.shuffle(self.indices)
        self.batch_size = batch_size

    def __next__(self):
        return self.__iter__()
                    
    def __call__(self):
        return self.__iter__()

    def make_batch(self, i):
        images = []
        labels = []
        for j in range(self.batch_size):
            image, label = self.pdata[self.indices[(i+j) % len(self.pdata)]]
            images += [image]
            labels += [label]
        images_tensor = torch.stack(images).resize_(self.batch_size, 1, 28*28)
        labels_tensor = torch.LongTensor(labels)
        return images_tensor, labels_tensor
        
    def __iter__(self):
        i = 0
        while True:
            images_tensor, labels_tensor = self.make_batch(i)
            i = i+self.batch_size
            if i >= len(self.pdata):
                np.random.shuffle(self.indices)
                i = i % len(self.pdata)
            yield [images_tensor, labels_tensor]
            

class DataFeeder(object):
    def __init__(self, data, preprocess_workers = 4, cuda_workers = 1, cpu_size = 12,
                 cuda_size = 3, batch_size = 12, use_cuda = True, volatile = False):
        self.preprocess_workers = preprocess_workers
        self.cuda_workers = cuda_workers
        self.use_cuda = use_cuda
        self.volatile = volatile
        
        self.train_batches_queue = Queue(maxsize=cpu_size)
        self.cuda_batches_queue = Queue(maxsize=cuda_size)

        #thread killers for ending threads
        self.train_thread_killer = thread_killer()
        self.train_thread_killer.set_tokill(False)
        self.cuda_thread_killer = thread_killer()
        self.cuda_thread_killer.set_tokill(False)

        #input generators
        self.input_gens = [BatchMaker(data, batch_size) for _ in range(preprocess_workers)]
        

    def start_queue_threads(self):
        for input_gen in self.input_gens:
            t = Thread(target=threaded_batches_feeder, args=(self.train_thread_killer, self.train_batches_queue, input_gen))
            t.start()
        for _ in range(self.cuda_workers):
            cudathread = Thread(target=threaded_cuda_batches, args=(self.cuda_thread_killer, self.cuda_batches_queue, self.train_batches_queue, self.use_cuda, self.volatile))
            cudathread.start()
            
    def kill_queue_threads(self):
        self.train_thread_killer.set_tokill(True)
        self.cuda_thread_killer.set_tokill(True)
        for _ in range(self.preprocess_workers):
            try:
                #Enforcing thread shutdown
                self.train_batches_queue.get(block=True,timeout=1)
            except Empty:
                pass
        for _ in range(self.cuda_workers):
            try:
                #Enforcing thread shutdown
                self.cuda_batches_queue.get(block=True,timeout=1)
            except Empty:
                pass

    def get_batch(self):
        return self.cuda_batches_queue.get(block=True)
    

if __name__ == "__main__":
    mnist_train = datasets.MNIST("/hdd/Data/MNIST/", train=True, transform=transforms.ToTensor(), download=True)
    mnist_val = datasets.MNIST("/hdd/Data/MNIST/", train=False, transform=transforms.ToTensor(), download=True)

    data_feeder = DataFeeder(mnist_train, preprocess_workers = 1, cuda_workers = 1, cpu_size = 12,
                 cuda_size = 3, batch_size = 12, use_cuda = True, volatile = False)
    data_feeder.start_queue_threads()

    i = 0
    while True:
        batch = data_feeder.get_batch()
        i += 1
        print(i)