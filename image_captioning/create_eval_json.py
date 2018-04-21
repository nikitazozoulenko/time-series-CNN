import json
from PIL import Image
from network import ImageAnnotator, GRUAnnotator
from lang import Lang

import torch
from torchvision import transforms
from torch.autograd import Variable

trfms = [transforms.Resize((224, 224))]
trfms += [transforms.ToTensor()]
trfms += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
pil_transforms = transforms.Compose(trfms)

def filename2cudaimage(file_path):
    im = Image.open(file_path).convert("RGB")
    tensor = pil_transforms(im)
    return Variable(tensor.cuda().unsqueeze(0), volatile=True)


def annotate_image(cuda_image, model):
    caption = model(cuda_image, None, test_time=True)
    return caption


def create_eval_json(coco_im_dir, info_path, model, out_file_name):
    with open(info_path) as json_data:
        file = json.load(json_data)
        for info in file:
            print(info)

        array_of_dicts = []
        for i, y in enumerate(file["images"]):
            if i % 100 == 0:
                print(i)
            ID = y["id"]
            filename = y["file_name"]
            cuda_image = filename2cudaimage(coco_im_dir+filename)
            caption = annotate_image(cuda_image, model)
            array_of_dicts += [{"image_id" : ID, "caption" : caption}]

    with open(out_file_name, 'w') as outfile:
        json.dump(array_of_dicts, outfile)


if __name__ == "__main__":
    print("Loading model...")
    lang = Lang()
    model = GRUAnnotator(embedding_size=512, hidden_size=512, n_layers=3, lang=lang).cuda()
    model.load_state_dict(torch.load("savedir/GRU1750k.pth"))
    model.eval()
    print("Successfully loaded model")
    
    coco_im_dir = "/hdd/Data/MSCOCO2015/images/val2014/"
    info_path = "/hdd/Data/MSCOCO2015/annotations/captions_val2014.json"
    create_eval_json(coco_im_dir, info_path, model, "captions_val2014_NikitaGRU_results.json")

    coco_im_dir = "/hdd/Data/MSCOCO2015/images/test2014/"
    info_path = "/hdd/Data/MSCOCO2015/annotations/image_info_test2014.json"
    create_eval_json(coco_im_dir, info_path, model, "captions_test2014_NikitaGRU_results.json")
