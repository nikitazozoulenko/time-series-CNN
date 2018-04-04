from torchvision import datasets
import re
import string
import os
import pickle

import torch
from torch.autograd import Variable
import numpy as np
  
class Lang():
    def __init__(self):
        self.EOS = 0
        self.UKN = 1
        self.EOS_string = "<EOS>"
        self.UKN_string = "<UKN>"

        if not os.path.exists("savedir/vocab/word2index.pkl"):
            if not os.path.exists("savedir/vocab/index2word.pkl"):
                self.create_dicts_from_scratch()

        with open('savedir/vocab/word2index.pkl', 'rb') as f:
            self.word2index = pickle.load(f)
        with open('savedir/vocab/index2word.pkl', 'rb') as f:
            self.index2word = pickle.load(f)

        self.vocab_size = len(self.word2index)


    def word2numpy(self, word):
        word = word.lower()
        word = re.sub('['+string.punctuation+']', '', word)
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.UKN

    
    def numpy2word(self, index):
        return self.index2word[index]

    
    def sentence2numpy(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub('['+string.punctuation+']', '', sentence).split()
        numpy_indices = [self.word2numpy(word) for word in sentence]
        return np.array(numpy_indices).astype(np.int64)

    
    def numpy2sentece(self, numpy_indices):
        caption = [self.numpy2word(index) for index in numpy_indices]
        caption[0] = caption[0].title()
        caption = " ".join(caption)
        return caption


    def sentence2variable(self, sentence, use_cuda = True, volatile = False):
        tensor = torch.from_numpy(self.sentence2numpy(sentence)).long()
        if use_cuda:
            tensor = tensor.cuda()
        return Variable(tensor, volatile=volatile)

    
    def variable2sentence(self, variable_sentence_indices):
        indices = variable_sentence_indices.data.cpu().numpy()
        return self.numpy2sentece(indices)


    def create_dicts_from_scratch(self):
        if not os.path.exists("savedir/vocab/count_dict.pkl"):
            print("word/count dict not found, generating new dict from scratch, takes approx 10 min")
            self.generate_vocabulary()

        count_dict = {}
        with open("savedir/vocab/count_dict.pkl", 'rb') as f:
            count_dict = pickle.load(f)

        word2index = {self.EOS_string : self.EOS, self.UKN_string : self.UKN}
        index2word = {self.EOS : self.EOS_string, self.UKN : self.UKN_string}
        count = len(word2index)
        for word, word_count in count_dict.items():
            if word_count >= 5:
                word2index[word] = count
                index2word[count] = word
                count += 1
        
        with open('savedir/vocab/word2index.pkl', 'wb') as f:
            pickle.dump(word2index, f)
        with open('savedir/vocab/index2word.pkl', 'wb') as f:
            pickle.dump(index2word, f)


    def generate_vocabulary(self):
        coco_path = "/hdd/Data/MSCOCO2017/images"
        annFile = "/hdd/Data/MSCOCO2017/annotations"
        data_train = datasets.CocoCaptions(root = coco_path+"/train2017/",
                                annFile = annFile+"/captions_train2017.json")
        # #only on train data, not val data
        #data_val = datasets.CocoCaptions(root = coco_path+"/val2017/",
        #                        annFile = annFile+"/captions_val2017.json")
        all_words = []
        count_dict = {}
        for data in data_train:
            im, captions = data
            for caption in captions:
                caption = caption.lower()
                caption = re.sub('['+string.punctuation+']', '', caption).split()
                for word in caption:
                    if word in all_words:
                        count_dict[word] += 1
                    else:
                        all_words.append(word)
                        count_dict[word] = 1

        with open('savedir/vocab/count_dict.pkl', 'wb') as f:
            pickle.dump(count_dict, f)


if __name__ == "__main__":
    lang = Lang()

    coco_path = "/hdd/Data/MSCOCO2017/images"
    annFile = "/hdd/Data/MSCOCO2017/annotations"
    data_val = datasets.CocoCaptions(root = coco_path+"/val2017/",
                                annFile = annFile+"/captions_val2017.json")
    for im, captions in data_val:
        for caption in captions:
            print(caption)
            caption = lang.sentence2variable(caption)
            print(caption)
            caption = lang.variable2sentence(caption)
            print(caption)
    



