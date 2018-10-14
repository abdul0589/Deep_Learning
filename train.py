import  matplotlib.pyplot as plt
import torch 
from torch import nn
from torch import optim
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
parser.add_argument('data_directory',help='Train directory')
parser.add_argument('-save_loc','--save_loc',help='directory to save checkpoints',default='checkpoint.pth',required=False)
parser.add_argument('-arch','--model_arch',help='choose a pretrained model architecture ;vgg13,vgg16 or densenet121',default="densenet121",required=False)
parser.add_argument('-learning_rate','--learn_rate',help='choose a lr for the model',default=0.001,type=float,required=False)
parser.add_argument('-hidden_units','--hidden_layers',help='choose hidden layers for the model',action='append',required=False)
parser.add_argument('-epochs','--n_epochs',help='choose the number of epochs for the model',default=5,required=False)
parser.add_argument('-gpu','--use_gpu',action='store_true',default=True,required=False)

args=parser.parse_args()
a_d=vars(args)

print (args)

data_dir=a_d['data_directory']
                    
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#Define transforms
data_transforms = transforms.Compose([transforms.RandomRotation(45),
                                     transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])              
                                    

# TODO: Load the datasets with ImageFolder
image_datasets_train = datasets.ImageFolder(train_dir,transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(image_datasets_train,batch_size=64,shuffle=True)

#Load validation directory
image_datasets_valid = datasets.ImageFolder(valid_dir,transform=data_transforms)

#Define the validation dataloaders
valid_dataloaders = torch.utils.data.DataLoader(image_datasets_valid,batch_size=64,shuffle=True)
                    
#model architecture
model_name=a_d['model_arch']
if model_name.lower()=='vgg16':
    model_name=models.vgg16(pretrained=True)
elif model_name.lower()=='vgg13':
    model_name=models.vgg13(pretrained=True)
elif model_name.lower()=='densenet121':
    model_name=models.densenet121(pretrained=True)

    
model=model_name

#get input size of the classifier layer of the model 
classifier_size=len(list(model.classifier.children()))

#find out the input size to the classifier 
if classifier_size>0:
    num_features=list(model.classifier.children())[0].in_features
else:
    num_features=model.classifier.in_features


#Build Network and attach the last layer of the pretrained model to the custom classifier 
if a_d['hidden_layers'] is  None:
    hidden_layers=[784]
    
else:
    hidden_layers=a_d['hidden_layers']

#define a network with custom architecture    
classifier=define_network.Network(num_features, 102, hidden_layers, drop_p=0.3)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

#attach custom classifier to pretrained model
model.classifier=classifier

#define criterion for optimizer
criterion=nn.NLLLoss()

#only train classifier parameters,feature parameters are frozen
optimizer=optim.Adam(model.classifier.parameters(),lr=a_d['learn_rate'])


if (a_d['use_gpu'] ==True) :
    enable_gpu=True
else:
    enable_gpu=False
    
device = torch.device("cuda:0" if enable_gpu else "cpu")    
model=model.to(device)
    
epochs = a_d['n_epochs']
steps = 0
running_loss = 0
print_every = 40
for e in range(epochs):
    model.train()
    for images, labels in train_dataloaders:
        steps += 1
        
        #convert images and labels to cuda types
        images, labels = images.to(device), labels.to(device)
                 

                    
        #zero the gradients 
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = utils.validation(model, valid_dataloaders, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(valid_dataloaders)),
                  "Validation Accuracy: {:.3f}".format(accuracy))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()                    

#save model class index                     
model.class_to_idx = image_datasets_train.class_to_idx

#extract and save the trained hidden layers to be used for reconstructing network later

trained_hidden_layers=hidden_layers



checkpoint = {
              'state_dict': model.state_dict(),
              'optim_dict':optimizer.state_dict(),
             'class_to_idx':model.class_to_idx,
             'number_of_epochs':epochs,
              'input_size': num_features,
              'hidden_layers': trained_hidden_layers,
             'output_size':102,
             'model_name':a_d['model_arch'] 
             }
           
 
torch.save(checkpoint, a_d['save_loc'])

print ("training completed successfully and model saved at {}".format(a_d['save_loc']))
