from PIL import Image
from torchvision import transforms as T
import numpy as np
import requests
import matplotlib.pyplot as plt
import cv2
import random

from urllib.request import urlopen
from io import BytesIO
from Dataset import COCO_INSTANCE_CATEGORY_NAMES
from model import model

def get_prediction(img_path, threshold=0.5, url=False):
    if url: # We have to request the image
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_path) # This is for local images
    transform = T.Compose([T.ToTensor()]) # Turn the image into a torch.tensor
    img = transform(img)
    img = img.cuda() # Only if GPU, otherwise comment this line
    pred = model([img]) # Send the image to the model. This runs on CPU, so its going to take time
  #Let's change it to GPU
  # pred = pred.cpu() # We will just send predictions back to CPU
  # Now we need to extract the bounding boxes and masks
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    resp = urlopen(url) # We want to convert URL to cv2 image here, so we can draw the mask and bounding boxes
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    return image

def random_color_masks(image):
    # I will copy a list of colors here
    colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image==1], g[image==1], b[image==1] = colors[random.randrange(0, 10)]
    colored_mask = np.stack([r,g,b], axis=2)
    return colored_mask