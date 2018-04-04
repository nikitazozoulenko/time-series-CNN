import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import cv2

from data_feeder import DataFeeder
from network import ImageAnnotator
from loss import Loss
from util_graphing import losses_to_ewma, graph, PredictionPreviewerReturner
from lang import Lang


def loop(out, canvas, model, ppr, skip_frames, ww, hh, nY, nX, b_size, widths, heights, val_data_feeder, use_cuda):
    ctr = 0
    while True:
        for j in range(nY):
            for i in range(nX):
                #reset canvas
                canvas[2*b_size+hh//nY*j:hh//nY*(j+1)-b_size, 2*b_size+ww//nX*i:ww//nX*(i+1)-b_size, :] = 0

                #get prediction
                images, _, _ = val_data_feeder.get_batch()
                caption = model(images, None, test_time=True)
                im = ppr(images)
                im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

                #DRAW IMAGE
                x = widths[i]
                y = heights[j]
                off_y = 70
                canvas[y-112-off_y:y+112-off_y,x-112:x+112,:] = im

                #DRAW CAPTION
                off_x = 350
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(canvas,"AI Prediction:",     (x-off_x,90+y), font, 0.7,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(canvas,caption,              (x-off_x,105+y), font, 0.5,(255,255,255),1,cv2.LINE_AA)

                #show frame
                for _ in range(skip_frames):
                    ctr+= 1
                    #cv2.imshow("Canvas", canvas)
                    out.write(canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q") or ctr == 20000:
                        return

def main():
    use_cuda = True
    coco_path = "/hdd/Data/MSCOCO2017/images"
    annFile = "/hdd/Data/MSCOCO2017/annotations"
    lang = Lang()
    val_data_feeder = DataFeeder(coco_path+"/val2017/",
                                   annFile+"/captions_val2017.json", 
                                   lang,
                                   preprocess_workers = 1, cuda_workers = 1, 
                                   cpu_size = 5, cuda_size = 2, 
                                   batch_size = 1, use_cuda = use_cuda, use_jitter = True, volatile = True)
    val_data_feeder.start_queue_threads()
    ppr = PredictionPreviewerReturner()
    model = ImageAnnotator(n_layers=18, hidden_size=256, lang=lang).cuda()
    model.load_state_dict(torch.load("savedir/model_01_it700k.pth"))
    model.eval()

    ww =1536
    hh = 864
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

    loop(out, canvas, model, ppr, skip_frames, ww, hh, nY, nX, b_size, widths, heights, val_data_feeder, use_cuda)
    
    cv2.destroyAllWindows()
    val_data_feeder.kill_queue_threads()

if __name__ == "__main__":
    main()
