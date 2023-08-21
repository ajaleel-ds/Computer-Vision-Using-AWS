import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse


#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass
    
def net():
    '''
    Initializes pre-trained model
    '''
    
    model = models.resnet50(pretrained=True) # Pulling in a pre-trained model

    for param in model.parameters():
        param.requires_grad = False   # Freezing convoluational layers
    
    num_features=model.fc.in_features # See the number of features present in the model, so we know how to add the Fully Connected layer at the end 
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model

def create_data_loaders(data, batch_size):
    
    # Dataset paths
    
    training_path = os.path.join(data, 'train')
    testing_path = os.path.join(data, 'test')
    validating_path = os.path.join(data, 'valid')
    
    
    # Configuring Image Transformers 
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Images standardzied to 224 X 224 for training. 
        transforms.RandomHorizontalFlip(), # To prevent potential possible biases in the training model.
        transforms.ToTensor(), # Converting to a tensor (a multi-dimensional array).
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizing the data -> image = (image - mean) / std
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Creating the Datasets
    
    train_data = torchvision.datasets.ImageFolder(
        root = training_path,
        transform = train_transform
    )
        
    test_data = torchvision.datasets.ImageFolder(
        root = testing_path,
        transform = test_transform
    )
    
    validation_data = torchvision.datasets.ImageFolder(
        root = validating_path,
        transform = test_transform,
    )
        
    # Data Loaders
        
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
    )   
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False,
    )
    
     validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size = batch_size,
        shuffle = False,
    )   
    
    
    return train_loader, test_loader, validation_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    
    args=parser.parse_args()
    
    main(args)
