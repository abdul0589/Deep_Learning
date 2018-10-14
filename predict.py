import  matplotlib.pyplot as plt
import torch 
from torch import nn
from torch import optim
import json
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from PIL import Image
import numpy as np
from collections import OrderedDict 
import pandas as pd
import argparse 
import define_network
import utils

parser=argparse.ArgumentParser()

#add arguments 
parser.add_argument('data_dir',help='Train directory')
parser.add_argument('-checkpoint_loc','--checkpoint',help='directory to save checkpoints',default='checkpoint.pth',required=False)
parser.add_argument('-top_k_predictions','--top_k',help='top k predictions',default=1,type=int,required=False)
parser.add_argument('-category_names','--categories',help='categories to map class predictions',default='cat_to_name.json',required=False)
parser.add_argument('-gpu','--use_gpu',action='store_true',help='choose gpu on/off',default=True,required=False)

args=parser.parse_args()
a_d=vars(args)

print (args)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    print (checkpoint.keys())        
    input_size=checkpoint['input_size']
    output_size=checkpoint['output_size']
    hidden_layers=checkpoint['hidden_layers']
    
    
    #pretrained model 
    model_name=checkpoint['model_name']
    if model_name.lower()=='vgg16':
        model=models.vgg16(pretrained=True)
    elif model_name.lower()=='vgg13':
        model=models.vgg13(pretrained=True)
    elif model_name.lower()=='densenet121':
        model=models.densenet121(pretrained=True)

    
    new_model_classifier = define_network.Network(input_size, output_size,hidden_layers , drop_p=0.3)
      
    model.classifier=new_model_classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.class_to_idx=checkpoint['class_to_idx']
       
    return model

the_model=load_checkpoint(a_d['checkpoint'])                   
print ("model loaded successfully")

if (a_d['use_gpu'] ==True) :
    enable_gpu=True
else:
    enable_gpu=False


device = torch.device("cuda:0" if enable_gpu else "cpu")    
model=the_model.to(device)


t1,t2=utils.predict(a_d['data_dir'],model,top_k=a_d['top_k'])

#convert cuda predictions to cpu
if enable_gpu==True:
    pred_classes=t2.cpu().numpy()
    pred_probs=t1.cpu().data.numpy()

elif enable_gpu==False:
    pred_classes=t2.numpy()
    pred_probs=t1.data.numpy()

filename=a_d['categories']

with open(filename, 'r') as f:
    cat_to_name = json.load(f)
    
#flatten 2 dimensional array into 1d
pred_classes=pred_classes.flatten('F')
pred_probs=pred_probs.flatten('F')


#invert dictionary
idx_to_class={k:v for v,k in the_model.class_to_idx.items()}

#map appropriate indexes to classes
top_k_indexes={x:(idx_to_class[z],y) for x,y in zip(pred_classes,pred_probs) for z in idx_to_class if x==z}


#map class names to indexes
top_k_classes={cat_to_name[y]:x[1] for x in top_k_indexes.values() for y in cat_to_name if x[0]==y}


for elements in top_k_classes:
    print (" The flower is {}  with a predicted probability of {}".format(elements,round(top_k_classes[elements]*100,2))) 