import argparse 
import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim 
from torchvision import datasets , transforms , models
import torch.nn.functional as F
from workspace_utils import keep_awake
from collections import OrderedDict
from fuctions import preprocess_data , choose_model , build_classifier ,train_model , test_model , save_model

parser = argparse.ArgumentParser()

parser.add_argument('data_dir',action = 'store' ,  help = 'choose the data directory' )
parser.add_argument('--save_dir',action = 'store' ,  help = 'choose the save directory' ,default='/home/workspace/ImageClassifier' )
parser.add_argument('--arch',action = 'store' ,  help = 'choose the model arch (densenet 121 , vgg11 )' ,default='densenet121' )
parser.add_argument('--lr',action = 'store' ,  help = 'choose learning rate for the model' ,default=0.002 , type = float )
parser.add_argument('--hidden_units',action = 'store' ,  help = 'choose number of hidden units for the classifier' ,default=512 ,type =int)
parser.add_argument('--epochs',action = 'store' ,  help = 'choose number of epochs' ,default=10 , type = int)
parser.add_argument('--gpu',action = 'store_true' ,  help = 'choose to enable gpu' ,default=False)

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
lr = args.lr
hidden_units = args.hidden_units
epochs =args.epochs
gpu = args.gpu

train_loader , valid_loader , test_loader , train_data= preprocess_data(data_dir)
model , input_units , device = choose_model(arch ,gpu)
model ,optimizer , criterion = build_classifier(model ,hidden_units , input_units , lr)
model = train_model(model,epochs ,lr ,train_loader ,valid_loader ,device ,optimizer ,criterion)
test_model(model , test_loader ,lr ,device,optimizer ,criterion)
save_model(model ,input_units ,train_data,optimizer, epochs,save_dir,arch )
