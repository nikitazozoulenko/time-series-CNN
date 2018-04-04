import os
import time
import threading
from threading import Thread
import sys
from queue import Empty,Full,Queue
import re
import string
import itertools
import random

import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import numpy as np

from lang import Lang


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
        for _, batch in enumerate(dataset_generator):
            #We fill the queue with new fetched batch until we reach the max size.
            batches_queue.put(batch, block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill,cuda_batches_queue,batches_queue, use_cuda, volatile):
    """Thread worker for transferring

unorm(tensor) pytorch tensors into
    GPU. batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        images_tensor, labels_tensor, num_words = batches_queue.get(block=True)

        if use_cuda:
            images_tensor = images_tensor.cuda()
            labels_tensor = labels_tensor.cuda()

        images_tensor = Variable(images_tensor, volatile = volatile)
        labels_tensor = Variable(labels_tensor, volatile = volatile)
        cuda_batches_queue.put((images_tensor, labels_tensor, num_words), block=True)
        
        if tokill() == True:
            return


class BatchMaker(object):
    def __init__(self, pdata, batch_size, lang):
        self.pdata = pdata
        self.lang = lang
        self.batch_size = batch_size

        self.indices = np.arange(len(pdata))
        np.random.shuffle(self.indices)


    def __next__(self):
        return self.__iter__()


    def __call__(self):
        return self.__iter__()


    def make_batch(self, i):
        batch_images = []
        batch_captions = []
        batch_num_words = []

        #extract data into lists
        for j in range(self.batch_size):
            image, captions = self.read_single_example(i, j)
            captions_indices = []
            num_words = []
            for caption in captions:
                indices = self.lang.sentence2numpy(caption)
                captions_indices += [indices]
                num_words += [len(indices)+1]

            #pick five out of the five/six/seven captions available per image
            sample = random.sample(range(len(captions_indices)), 5)
            final_five_captions = []
            final_five_lengths = []
            for l, ex in enumerate(captions_indices):
                if l in sample:
                    final_five_captions += [captions_indices[l]]
                    final_five_lengths += [num_words[l]]

            batch_images += [image]
            batch_captions += [final_five_captions]
            batch_num_words += [final_five_lengths]
        
        #make lists into real arrays
        max_words = max(itertools.chain(*batch_num_words))
        for n, captions in enumerate(batch_captions):
            for m, caption in enumerate(captions):
                for _ in range(len(caption), max_words):
                    batch_captions[n][m] = np.append(batch_captions[n][m], self.lang.EOS)

        #to tensors
        numpy_captions = np.array(batch_captions)
        numpy_num_words = np.array(batch_num_words)
        tensor_images = torch.stack(batch_images)
        tensor_captions = torch.from_numpy(numpy_captions)

        return tensor_images, tensor_captions, numpy_num_words


    def read_single_example(self, i, j):
        index = self.indices[(i+j) % len(self.pdata)]
        image, caption = self.pdata[index]
        return image, caption


    def __iter__(self):
        i = 0
        while True:
            images_tensor, labels_tensor, num_words_tensor = self.make_batch(i)
            i = i+self.batch_size
            if i >= len(self.pdata):
                np.random.shuffle(self.indices)
                i = i % len(self.pdata)
            yield [images_tensor, labels_tensor, num_words_tensor]
            

class DataFeeder(object):
    def __init__(self, coco_path, annFile, lang, preprocess_workers = 4, cuda_workers = 1, cpu_size = 12, cuda_size = 3, batch_size = 12, use_jitter = False, use_cuda = True, volatile = False):
        trfms = [transforms.Resize((224, 224))]
        if use_jitter:
            trfms += [transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)]
        trfms += [transforms.RandomHorizontalFlip()]
        trfms += [transforms.ToTensor()]
        trfms += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        pil_transforms = transforms.Compose(trfms)
        data = datasets.CocoCaptions(root = coco_path, annFile = annFile, transform = pil_transforms)

        self.lang = lang
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
        self.input_gens = [BatchMaker(data, batch_size, self.lang) for _ in range(preprocess_workers)]
        

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
    coco_path = "/hdd/Data/MSCOCO2017/images"
    annFile = "/hdd/Data/MSCOCO2017/annotations"
    use_cuda = True

    lang = Lang()
    train_data_feeder = DataFeeder(coco_path+"/train2017/",
                                   annFile+"/captions_train2017.json",
                                   lang, 
                                   preprocess_workers = 4, cuda_workers = 1, 
                                   cpu_size = 15, cuda_size = 3, 
                                   batch_size = 3, use_cuda = use_cuda, use_jitter = True, volatile = False)

    train_data_feeder.start_queue_threads()


    for i in range(100):
        batch = train_data_feeder.get_batch()
        images, captions, num_words = batch
        print(num_words)
    train_data_feeder.kill_queue_threads()