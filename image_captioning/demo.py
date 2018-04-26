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


def loop(out, canvas, TCN, GRU, ppr, skip_frames, ww, hh, nY, nX, b_size, widths, heights, feeder, use_cuda):
    ctr = 0
    while True:
        for j in range(nY):
            for i in range(nX):
                #reset canvas
                canvas[2*b_size+hh//nY*j:hh//nY*(j+1)-b_size, 2*b_size+ww//nX*i:ww//nX*(i+1)-b_size, :] = 0

                #get prediction
                show_image, var_image = feeder.get_batch()
                GRUcaption = GRU(var_image, None, test_time=True)
                TCNcaption = TCN(var_image, None, test_time=True)
                im = cv2.cvtColor(np.asarray(show_image), cv2.COLOR_RGB2BGR)

                #DRAW IMAGE
                x = widths[i]
                y = heights[j]
                off_y = 70
                canvas[y-112-off_y:y+112-off_y,x-112:x+112,:] = im

                #DRAW CAPTION
                off_x = 350
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(canvas,"TCN Prediction:",     (x-off_x,90+y), font, 0.7,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(canvas,TCNcaption,            (x-off_x,105+y), font, 0.5,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(canvas,"GRU Prediction:",     (x-off_x,150+y), font, 0.7,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(canvas,GRUcaption,            (x-off_x,165+y), font, 0.5,(255,255,255),1,cv2.LINE_AA)

                #show frame
                for _ in range(skip_frames):
                    ctr+= 1
                    #cv2.imshow("Canvas", canvas)
                    out.write(canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q") or ctr == 20000:
                        return

def main():
    use_cuda = True
    coco_path = "/hdd/Data/MSCOCO2017/images/val2017/"
    annFile = "/hdd/Data/MSCOCO2017/annotations/captions_val2017.json"
    feeder = DemoImageFeeder(coco_path, annFile, show_size=224, use_cuda=use_cuda)
    lang = Lang()
    ppr = PredictionPreviewerReturner()
    TCN = ImageAnnotator(n_layers=18, hidden_size=256, lang=lang).cuda()
    TCN.load_state_dict(torch.load("savedir/TCN1750k.pth"))
    TCN.eval()

    GRU = GRUAnnotator(embedding_size=512, hidden_size=512, n_layers=3, lang=lang).cuda()
    GRU.load_state_dict(torch.load("savedir/GRU1750k.pth"))
    GRU.eval()

    ww =1920
    hh = 1080
    nX = 2
    nY = 2
    skip_frames = 40
    b_size = 5

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("video-caption-demo.avi", fourcc, 20.0, (ww, hh))

    boxwidth = ww/nX
    boxheight = hh/2
    widths = [int((i+0.5)*boxwidth) for i in range(nX)]
    heights = [int((i+0.5)*boxheight) for i in range(2)]

    canvas = np.ones((hh,ww,3)).astype(np.uint8) * 255
    canvas[2*b_size:-2*b_size,2*b_size:-2*b_size,:] = 0

    for k in range(nY):
        canvas[hh//nY*k-b_size:hh//nY*k+b_size,:,:] = 255
    for k in range(nX):
        canvas[:,ww//nX*k-b_size:ww//nX*k+b_size,:] = 255

    loop(out, canvas, TCN, GRU, ppr, skip_frames, ww, hh, nY, nX, b_size, widths, heights, feeder, use_cuda)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
