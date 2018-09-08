import argparse

import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F

import numpy as np

from PIL import Image

import json
import os
import random

from utils import load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg') # use a deafault filepath to a primrose image 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img_pil = Image.open(image) # use Image
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = adjustments(img_pil)
    
    return image

def predict(image_path, model, topk=3, gpu='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
        
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1) # use F 
    
    return probability.topk(topk)

def main(): 
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    
    img_path = args.filepath
    probabilities = predict(img_path, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    print('File selected: ' + img_path)
    
    print(labels)
    print(probability)
    
    i=0 # this prints out top k classes and probs as according to user 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 # cycle through

if __name__ == "__main__":
    main()