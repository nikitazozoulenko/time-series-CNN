import json
from network import ImageAnnotator, GRUAnnotator

def annotate_image_from_file(filename):
    return "dummy string annotation"


def create_eval_json(path_to_ids):
    with open(path_to_ids) as json_data:
        file = json.load(json_data)
        for info in file:
            print(info)

        array_of_dicts = []
        for i, y in enumerate(file["images"]):
            print(i)
            ID = y["id"]
            filename = y["file_name"]
            caption = annotate_image_from_file(filename)
            array_of_dicts += [{"image_id" : ID, "caption" : caption}]

    with open('demo/val_results.json', 'w') as outfile:
        json.dump(array_of_dicts, outfile)


if __name__ == "__main__":
    #create_eval_json("/hdd/Downloads/annotations/image_info_test2017.json")
    create_eval_json("/hdd/Data/MSCOCO2017/annotations/captions_val2017.json")
