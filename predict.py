import argparse 
import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim 
from torchvision import datasets , transforms , models
from PIL import Image
import torch.nn.functional as F
from workspace_utils import keep_awake
import json
from collections import OrderedDict
from fuctions import load_checkpoint , process_image , predict_image
parser = argparse.ArgumentParser()

parser.add_argument('--img_pth' , action = 'store' , help = 'choose image path', default = 'flowers/test/50/image_06297' )
parser.add_argument('--top_k' , action = 'store' , help = 'choose number of top classes', default = 5 , type =int)
parser.add_argument('--category_names' , action = 'store' , help = 'choose category names file', default = 'cat_to_name.json' )
parser.add_argument('--gpu',action = 'store_true' ,  help = 'choose to enable gpu' ,default=False)
parser.add_argument('--checkpoint' , action = 'store' , help = 'choose checkpoint file', default = 'checkpoint.pth' )

args = parser.parse_args()

img_path = args.img_pth
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu
checkpoint = args.checkpoint

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
    
model = load_checkpoint(checkpoint)

if gpu == True:
   device = 'cuda'
else:
   device = 'cpu'

probs,classes = predict_image(img_path ,model , top_k, device ,cat_to_name )
print(probs)
print(classes)