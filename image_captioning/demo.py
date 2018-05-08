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
        self.index = np.arange(len(self.data))
        np.random.shuffle(self.index)
        self.show_size = show_size
        self.use_cuda = use_cuda
        
        trfms = [transforms.Resize((224, 224))] 
        trfms += [transforms.ToTensor()]
        trfms += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transforms = transforms.Compose(trfms)


    def get_batch(self):
        #returns showimage, modelimage
        image, caption = self.data[self.index[self.n]]
        self.n = (self.n + 1) % len(self.data)
        tensor_im = self.transforms(image).unsqueeze(0)
        if self.use_cuda:
            tensor_im = tensor_im.cuda()
        var_im = Variable(tensor_im, volatile=True)
        return image.resize((self.show_size, self.show_size)), var_im


def reformat_caption(caption):
    per_row = 23
    final = []
    while(len(caption)>0):
        if len(caption) > per_row:
            index = caption.rfind(" ", 0, per_row)
            final += [caption[0:index]+"\n"]
            caption = caption[index:]
            if caption[0] == " ":
                caption = caption[1:]
        else:
            final += [caption[:]]
            caption = ""

    out = "".join(final)
    return out


def draw_caption(canvas, caption, title, offset, x, y, startX, startY):
    off_x = 500
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas,title,     (off_x+x*startX,offset+60+y*startY), font, 1.7,(255,255,255),2,cv2.LINE_AA)
    y0, dy = 100, 30
    for i, line in enumerate(caption.split('\n')):
        yyy = y0 + i*dy
        cv2.putText(canvas, line, (off_x+x*startX,offset+yyy+y*startY), font, 1.1,(255,255,255),1,cv2.LINE_AA)


def draw_im_on_canvas(canvas, im, GRUcaption, TCNcaption, border_size, image_size, x, y):
    smallSize = True
    if smallSize:
        im = im.resize((224,224)).resize((image_size, image_size))
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    startX = 1920//2-border_size//2
    startY = image_size+border_size
    #image
    canvas[y*startY+border_size:y*startY+image_size+border_size, x*startX+border_size:x*startX+image_size+border_size, :] = im
    #black area
    canvas[y*startY+border_size:y*startY+image_size+border_size,x*startX+2*border_size+image_size:x*startX+1920//2-border_size//2,:] = 0

    #DRAW CAPTION
    draw_caption(canvas, TCNcaption, "TCN Prediction:", 0, x, y, startX, startY)
    draw_caption(canvas, GRUcaption, "GRU Prediction:", 235, x, y, startX, startY)


def loop(out, canvas, TCN, GRU, skip_frames, feeder, border_size, image_size):
    ctr = 0
    while True:
        for j in range(2):
            for i in range(2):
                #get prediction
                im, var_image = feeder.get_batch()
                GRUcaption = GRU(var_image, None, test_time=True)
                TCNcaption = TCN(var_image, None, test_time=True)
                GRUcaption = reformat_caption(GRUcaption)
                TCNcaption = reformat_caption(TCNcaption)

                #DRAW IMAGE and caption
                draw_im_on_canvas(canvas, im, GRUcaption, TCNcaption, border_size, image_size, j, i)

                #show frame
                for _ in range(skip_frames):
                    ctr+= 1
                    cv2.imshow("Canvas", canvas)
                    out.write(canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q") or ctr == 1000:
                        return


def main():
    skip_frames = 40
    border_size = 10
    image_size = 470

    use_cuda = True
    coco_path = "/hdd/Data/MSCOCO2017/images/val2017/"
    annFile = "/hdd/Data/MSCOCO2017/annotations/captions_val2017.json"
    feeder = DemoImageFeeder(coco_path, annFile, show_size=image_size, use_cuda=use_cuda)
    lang = Lang()
    ppr = PredictionPreviewerReturner()
    TCN = ImageAnnotator(n_layers=18, hidden_size=256, lang=lang).cuda()
    TCN.load_state_dict(torch.load("savedir/TCN1750k.pth"))
    TCN.eval()

    GRU = GRUAnnotator(embedding_size=512, hidden_size=512, n_layers=3, lang=lang).cuda()
    GRU.load_state_dict(torch.load("savedir/GRU1750k.pth"))
    GRU.eval()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("video-caption-demo.avi", fourcc, 20.0, (1920, 1080))

    canvas = np.ones((1080,1920,3)).astype(np.uint8)*255
    canvas[3*border_size+2*image_size:1080-border_size, border_size:-border_size, :] = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas,"The images were taken from the Microsoft Common Objects in Context (MSCOCO) validation dataset", (50, 1080-50), font, 1 ,(255,255,255),1,cv2.LINE_AA)

    #loop(out, canvas, TCN, GRU, ppr, skip_frames, ww, hh, nY, nX, b_size, widths, heights, feeder, use_cuda)
    loop(out, canvas, TCN, GRU, skip_frames, feeder, border_size, image_size)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
