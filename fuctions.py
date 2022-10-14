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

def preprocess_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir , transform = train_transforms)
    validation_data = datasets.ImageFolder(valid_dir , transform = validation_transforms)
    test_data = datasets.ImageFolder(test_dir , transform = test_transforms)
    # Using the image datasets and the trainforms, define the dataloaders

    train_loader = torch.utils.data.DataLoader(train_data , batch_size=64 ,shuffle = True)
    valid_loader = torch.utils.data.DataLoader(validation_data , batch_size=64 )
    test_loader = torch.utils.data.DataLoader(test_data , batch_size=64 )
    
    return train_loader , valid_loader , test_loader , train_data



def choose_model(arch , gpu):
    
    
    if arch=='densenet121':
        model = models.densenet121(pretrained=True)
        input_units = model.classifier.in_features
        print("using densenet121")
    else:
        print("densenet121 or vgg16 only : using vgg16")
        model =models.vgg16(pretrained=True) 
        input_units = model.classifier[0].in_features
        
    if gpu == True:
        device = 'cuda'
    else:
        device = 'cpu'
        
    return model , input_units , device


def build_classifier(model ,hidden_units , input_units ,lr):
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    model.classifier = nn.Sequential(nn.Dropout(0.1),
                                    nn.Linear(input_units,hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512,102),
                                    nn.LogSoftmax(dim=1))
    return model , optimizer , criterion


def train_model(model,epochs , lr , train_loader , valid_loader ,device ,optimizer ,criterion):
        
    model.to(device); 
    running_loss = 0
    steps  = 0
    print_var  = 30
    for epoch in keep_awake(range(epochs)):
        for images , labels in train_loader:
            optimizer.zero_grad()
            images , labels = images.to(device) , labels.to(device)
            steps+=1 

            logps = model(images)
            loss = criterion(logps ,labels)
            loss.backward()
            optimizer.step()
            running_loss+= loss.item()

            if steps % print_var == 0:
                accuracy = 0
                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for images , labels in valid_loader:
                        images , labels = images.to(device) , labels.to(device)

                        logps= model(images)
                        batch_loss = criterion(logps, labels)
                        val_loss+= batch_loss
                        ps = torch.exp(logps)

                        top_p , top_class  = ps.topk(1,dim =1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"epoch: {epoch+1}/{epochs}.. "
                          f"training_loss: {running_loss/print_var:.3f}.. "
                          f"val_loss: {val_loss/len(valid_loader):.3f}.. "
                          f"val_accuracy: {accuracy/len(valid_loader):.3f}")
                    running_loss = 0
                    model.train()
    print("done trainig")
    return model


def test_model(model , test_loader  ,lr , device ,optimizer ,criterion ):
    
    model.to(device);
    model.eval()
    test_loss = 0
    test_accuracy = 0 
    with torch.no_grad():
        for images , labels in test_loader:
            optimizer.zero_grad()
            images , labels = images.to(device) , labels.to(device)           
            logps= model(images)
            b_loss = criterion(logps, labels)
            test_loss+= b_loss
            ps = torch.exp(logps)
            top_p , top_class  = ps.topk(1,dim =1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"test accuracy: {(test_accuracy/len(test_loader))*100:.2f} %")
    
def save_model(model , input_units , train_data, optimizer , epochs , save_dir , arch):
 
    checkpoint = {'input_size': input_units,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'arch':arch,
                  'class_to_idx':  train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,   
                  'num_epochs': epochs,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir+'/checkpoint.pth')
    
    return checkpoint

#----------------------------------------------

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch']=='densenet121':
        model = models.densenet121(pretrained=True)
    else:
        model =models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img = Image.open(f"{image}.jpg")
    
    process = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
    pil_image = process(img)
    np_image = np.array(pil_image)
    
    return np_image

def predict_image(image_path, model, topk ,device ,cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze(0)
    img = img.to(device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        logps = model(img)
        ps = torch.exp(logps)
        probs , indexs = ps.topk(topk)
    indexs,probs  = np.array(indexs) ,np.array(probs)
    #inverting class dict
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    top_classes = []
    
    for i in indexs[0]:
        top_classes.append(idx_to_class[i])
        
    names = [cat_to_name[i] for i in top_classes]

    return probs.tolist() , names