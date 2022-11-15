#!/usr/bin/env python3
#                                                                             
# PROGRAMMER: Htet Wai Yan
##
# Imports python modules
import argparse

def get_input_args_for_train():

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, 
                        help='path to data directory that contains test, train, valid sub folders. just type the folder name only.')
    parser.add_argument('--gpu', 
                        help = 'use GPU for training the model', action='store_true')
    parser.add_argument('--save_dir', type = str,
                    help = 'path to save the checkpoint') 
    parser.add_argument('--arch', type = str, default = 'alexnet', 
                    help = 'Classification algorithm')
    parser.add_argument('--learning_rate', type = float, default = '0.001', 
                    help = 'Learning rate for the model') 
    parser.add_argument('--hidden_units', type = int, default = '512', 
                    help = 'Hidden units for the classifier') 
    parser.add_argument('--epochs', type = int, default = '10', 
                    help = 'Number of epochs')

 
    return parser.parse_args()

def get_input_args_for_pred():
     # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('img_path', type=str, 
                        help='path to a single image.')
    parser.add_argument('checkpoint', type=str, 
                        help='The checkpoint file for the model')
    parser.add_argument('--gpu', 
                        help = 'use GPU for prediction', action='store_true')
    parser.add_argument('--top_k', type = int, default=5,
                    help = 'K mostly class of the image prediction') 
    parser.add_argument('--category_names', type = str,
                    help = 'Category names json file')

 
    return parser.parse_args()