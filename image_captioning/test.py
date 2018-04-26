import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import cv2

from torchvision import datasets, transforms

from data_feeder import DataFeeder
from network import ImageAnnotator, GRUAnnotator
from loss import Loss
from util_graphing import losses_to_ewma, PredictionPreviewerReturner
from lang import Lang

class DemoImageFeeder():
    def __init__(self, coco_path, annFile, show_size=500, model_input_size=224, use_cuda = True):
        self.data = datasets.CocoCaptions(root = coco_path, annFile = annFile)
        self.n = 0
        self.show_size = show_size
        self.use_cuda = use_cuda
        
        trfms = [transforms.Resize((224, 224))] 
        trfms += [transforms.ToTensor()]
        trfms += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transforms = transforms.Compose(trfms)

    def get_batch(self):
        #returns showimage, modelimage
        image, caption = self.data[self.n]
        self.n = (self.n + 1) % len(self.data)
        tensor_im = self.transforms(image).unsqueeze(0)
        if self.use_cuda:
            tensor_im = tensor_im.cuda()
        var_im = Variable(tensor_im, volatile=True)
        return image.resize((self.show_size, self.show_size)), var_im

if __name__ == "__main__":
    coco_path = "/hdd/Data/MSCOCO2017/images/val2017/"
    annFile = "/hdd/Data/MSCOCO2017/annotations/captions_val2017.json"
    feeder = DemoImageFeeder(coco_path, annFile, show_size=500)
    im = feeder.get_batch()
    print(im)