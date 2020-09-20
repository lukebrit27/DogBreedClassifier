import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torchvision.models as models
import os
from torchvision import datasets
from PIL import ImageFile
import torch.nn as nn
from glob import glob
from tqdm import tqdm
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
from workspace_utils import keep_awake
from train_funcs import *


def create_network_nodes(in_features, no_labels, no_hidden_layers, no_hidden_nodes, node_ratio):

    total_layers = no_hidden_layers + 2 #hidden layers + input layer + output layer
    network_nodes = [in_features]
    unassigned_nodes = no_hidden_nodes #nodes not assigned to a layer yet. Assigned in for loop below

    #assign hidden nodes to hidden layers
    for i in range(no_hidden_layers):
        if i < (no_hidden_layers - 1):
            network_nodes.append( int(np.ceil(node_ratio*unassigned_nodes)) )
        else:
            network_nodes.append(unassigned_nodes)

        unassigned_nodes -= network_nodes[-1]

    #append no. of output nodes
    network_nodes.append(no_labels)
    print('unassigned_nodes = ', unassigned_nodes)

    print('Number of input features: ',in_features)
    print('Number of Labels: ',no_labels)
    print('Number of Hidden Layers: ',no_hidden_layers)
    print('Total Number of Layers: ',total_layers)
    print('Number of Hidden Nodes: ',no_hidden_nodes)
    print('Breakdown of No. of Nodes for each layer in our Network: ', network_nodes)

    return network_nodes


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    outF = open("myOutFile.txt", "w")
    train_loss = 0.0
    valid_loss = 0.0
    train_accuracy = 0.0

    for epoch in keep_awake(range(1, n_epochs+1)):


        ###################
        # train the model #
        ###################

        #model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly

            #get rid of gradients from last weight update
            optimizer.zero_grad()

            log_probs = model.forward(data)
            loss = criterion(log_probs,target)
            loss.backward()
            optimizer.step()

            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

             # Calculate accuracy
            #probs = torch.exp(log_probs)
            probs = F.softmax(log_probs, dim = 1)
            top_prob, top_class = probs.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


        ######################
        # validate the model #
        ######################
        valid_accuracy = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## update the average validation loss
                log_probs = model.forward(data)
                loss = criterion(log_probs,target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))

                 # Calculate accuracy
                #probs = torch.exp(log_probs)
                probs = F.softmax(log_probs, dim = 1)
                top_prob, top_class = probs.topk(1, dim=1)
                equals = top_class == target.view(*top_class.shape)
                valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # print training/validation statistics
        train_accuracy = train_accuracy/len(loaders['train'])
        valid_accuracy = valid_accuracy/len(loaders['valid'])
        log_text='Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTrain Accuracy {:.6f} \tValidation Accuracy {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss,
            train_accuracy,
            valid_accuracy
            )
        print(log_text)
        outF.write(log_text)
        outF.write("\n")


        model.train()


        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            print('validation loss has decreased, saving current state of model for epoch {}'.format(epoch))
            torch.save(model.state_dict(), save_path)

        # reset variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_accuracy = 0.0


    # return trained model
    outF.close()
    return model

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()


        network_nodes = create_network_nodes(25088,133,3,2048,0.5)
        for i in range(len(network_nodes) - 1):
            prog = 'self.fc{} = nn.Linear({}, {})'.format(i+1,network_nodes[i],network_nodes[i+1])
            exec(prog)

       # Dropout module drop probability
        self.network_nodes = network_nodes
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        network_nodes = self.network_nodes
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        #print(x)
        # Now with dropout
        for i in range(len(network_nodes) - 1):
            if i < len(network_nodes) - 2:
                prog = 'self.dropout(F.relu(self.fc{}(x)))'.format(i+1)
            else:
                prog = 'F.log_softmax(self.fc{}(x), dim=1)'.format(i+1)

            x = eval(prog)
        #print(x)
        # output so no dropout here
        #x = F.log_softmax(self.fc4(x), dim=1)

        return x
