# vqa_variants

Here, we implement a variant of the "A+C+S-K-LSTM" model as described in the paper [Image Captioning and Visual Question Answering Based on Attributes and External Knowledge](https://arxiv.org/pdf/1603.02814.pdf) by Q Wu et al.

The VQA model is a CNN-RNN model that takes an input image and input question and generates an output response.

Attribute-based Image Representation -> V_att
Image Captioning -> V_cap
External Knowledge Base -> V_knowledge

## Attribute-based Image Representation

### Setup

We adapt YOLO v3 to extract the image attributes. To download YOLO v3, run:
```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
And then run this to download the pre-trained weights:
```
wget https://pjreddie.com/media/files/yolov3.weights
```

We use the dataset from train-2014 MS COCO. Download by running this in root directory:
```
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
mkdir coco-train2014
gsutil -m rsync gs://images.cocodataset.org/train2014 coco-train2014
```

In the root directory run:
```
mkdir Vatt
```
Finally, download the `modified_darknet.py` file and put it in `darknet/python/`, and the `query.py` file and put it into `Vatt`

### Run

In the directory `darknet`, run:
```
python python/modified_darknet.py
```

This will run YOLOv3 on all the train-2014 MS COCO images, and produce 2 JSON dumps in the `Vatt` folder, `Vatt.json` and `top5.json`.

## Knowledge Based Representation

In the directory `Vatt`, run:
```
python query.py
```

This will query the knowledge database with the top 5 attributes in the picture, concatenate the results into one long paragraph to be passed into Doc2Vec.




