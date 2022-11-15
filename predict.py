from get_input_args import get_input_args_for_pred

import os
import torch
from torchvision import models

from PIL import Image

import numpy as np

import json

def main():
    # Getting the command line arguments
    print('Getting Commanding Line Arguments ...')
    in_arg = get_input_args_for_pred()
    # print(in_arg)
    # Get the values from the command line arguments
    img_path = in_arg.img_path
    checkpoint = in_arg.checkpoint
    use_gpu = in_arg.gpu
    top_k = in_arg.top_k
    category_names = in_arg.category_names
    
    # Check the image file exists
    if not os.path.exists(img_path):
        raise Exception('Image file does not exist.');
        
    # Check the checkpoint file exists
    if not os.path.exists(checkpoint):
        raise Exception('Checkpiont file does not exist.');
    
    # Check the category names file exists
    if category_names is not None and not os.path.exists(category_names):
        raise Exception('Category name file does not exist.');
    
    device = 'cpu'

    # Check GUP Available if the user wants to use GPU
    if use_gpu:
        is_gpu_available = torch.cuda.is_available()
        if is_gpu_available:
            device = 'cuda'
            print('Running in GPU mode ...')
        else:
            print('GPU unavailable. Running in CPU mode ....')
    else:
        print('Running in CPU mode ...')
    model = load_check_point(checkpoint)
    model.to(device)
    
    prob, classes = predict(model, img_path, top_k, device)
    print_results(prob, classes, category_names)
    
def load_check_point(checkpoint):
    checkpoint = torch.load(checkpoint)
    arch = checkpoint['arch']
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    return model

# Process the image to be able to use as an input for the model
def process_image(image):

    image = Image.open(image)
    
    if image.width > image.height:
        image.thumbnail((10000000, 256))
    else:
        image.thumbnail((256, 10000000))
    
    
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = (image.width + 224) / 2
    bottom = (image.height + 224) / 2
    
    image = image.crop((left, top, right, bottom))
    
    image = np.array(image)
    image = image / 255

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    image = image.transpose(2, 0, 1)
    return image

def predict(model, image_path, topk, device):
    # set the model in inference mode
    model.eval()
    
    with torch.no_grad():
        image = process_image(image_path)
        img = torch.from_numpy(np.asarray(image).astype('float'))
        image = img.unsqueeze_(0)
        image = img.float()
        image = image.to(device)
        outputs = model.forward(image)
        probs, classes = torch.exp(outputs).topk(topk, dim=1)
        inverse_index = {model.class_to_idx[i]: i for i in model.class_to_idx}
        fclasses = list()
    
        for c in classes.tolist()[0]:
            fclasses.append(inverse_index[c])
        
        return probs[0].tolist(), fclasses
    
def print_results(probabilities, classes, category_names):
    cat_to_name = None
    if category_names is not None:
        try:
            with open(category_names, 'r') as f:
                cat_to_name = json.load(f)
                print(cat_to_name)
        except:
            raise Exception('Category name file is not valid')
    i = 0
    print('---Results---')
    for prob, label in zip(probabilities, classes):
        i = i + 1
        if (cat_to_name):
            label = cat_to_name[label]
        else:
            label = 'Class {}'.format(str(label))
        print(f"{i}. {label} [{(prob*100):.3f}%]")
    return None
            
        
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
