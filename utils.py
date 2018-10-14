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


# Implement a function for the validation 
def validation(model, testloader, criterion):

    test_loss = 0
    accuracy = 0
    correct = 0
    total = 0
    for images, labels in testloader:

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        _, predicted = torch.max(ps.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy=(100 * correct / total)
    
    return test_loss, accuracy
 




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image=Image.open(image)
    crop_size=(224,224)
    

    largest=pil_image.size[1]
    shortest=pil_image.size[0]

    pil_image.thumbnail((256,largest))
    
    
    
    awidth, aheight = pil_image.size
    bwidth, bheight = crop_size

    l = (awidth - bwidth)/2
    t = (aheight - bheight)/2
    r = (awidth + bwidth)/2
    b = (aheight + bheight)/2

    pil_image=pil_image.crop((l, t, r, b))

    np_image = np.array(pil_image)
    np_image=np_image/255

    mean=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])

    np_image=np_image-mean
    np_image=np_image/std

    np_image=np_image.transpose(2,1,0)
     
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose(1, 2, 0)
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    processed_image=process_image(image_path)
    
    image_tensor = torch.from_numpy(processed_image).type(torch.FloatTensor)
    image = image_tensor.unsqueeze_(0)#this adds another dimension to the image to match tensor dimension
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

    image=image.to(device)
    model=model.to(device)
    model.eval()
        
    outputs=model(image)
    ps = torch.exp(outputs)
    predicted = ps.topk(top_k)
    
    
    return predicted
