from pycocotools.coco import COCO
import json
dataDir='.'
dataType='train2014'
annFile = './annotations/captions_train2014.json'.format(dataDir,dataType)
coco=COCO(annFile)

captions = {}


with open('./annotations/captions_train2014.json', 'r') as f:
    data = json.load(f)
    for img in data["images"]:
        img_name = img['file_name']
        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)
        ann = coco.showAnns(anns)
        captions[img_name] = ann

with open('./captions_train2014.json', 'w') as f:
    json.dump(captions, f)
