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

# load filenames for human and dog images
human_files = np.array(glob("data/lfw/*/*"))
dog_files = np.array(glob("data/dog_images/*/*/*"))

data_dir = 'data/dog_images'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# define VGG16 model, used for dog detector
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()

#load trained model
model_transfer = models.vgg16(pretrained=True)
for param in model_transfer.parameters():
    param.requires_grad = False
if use_cuda:
    device = 'cuda:0'
else:
    device = 'cpu'
model_transfer.load_state_dict(torch.load('model_transfer.pt',map_location=device))


#get list of possible dog breeds from training data
train_transforms = transforms.Compose([transforms.RandomRotation(30), #rotate 30 degrees random direction
                                       #transforms.RandomResizedCrop(224),#randomly resize and take 224 centre crop
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],#means of each color channel
                                                            [0.229, 0.224, 0.225])])#sds of each color channel
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in train_data.classes]


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    img = Image.open(img_path)
    img_transforms = transforms.Compose([transforms.Resize(256),#resize 256x256 pixels(squares)
                                       transforms.CenterCrop(224),#crops square from center with 224 pixels on each side
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    transformed_img = img_transforms(img)

    VGG16.eval()
    res = VGG16.forward(transformed_img.reshape(1,3,224,224))

    # call softmax to get probabilities from linear result
    probs = F.softmax(res[0])
    category_pred = np.where(probs == max(probs))[0][0]


    return category_pred # predicted class index

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):

    pred = VGG16_predict(img_path)

    return pred >= 151 and pred <= 268 # true/false

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed

    img = Image.open(img_path)
    img_transforms = transforms.Compose([transforms.Resize(256),#resize 256x256 pixels(squares)
                                       transforms.CenterCrop(224),#crops square from center with 224 pixels on each side
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    transformed_img = img_transforms(img)

    model_transfer.eval()
    log_probs = model_transfer.forward(transformed_img.reshape(1,3,224,224))
    probs = torch.exp(log_probs)
    top_prob, top_class = probs.topk(1, dim=1)
    #equals = top_class == labels.view(*top_class.shape)

    return class_names[top_class]


def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    if dog_detector(img_path):
        print('Hello there doggy!')
        img = mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.show()
        print('You like a ...')
        pred = predict_breed_transfer(img_path)
        print(pred)


    elif face_detector(img_path):
        print('Hello there human!')
        img = mpimg.imread(img_path)
        imgplot = plt.imshow(img)
        plt.show()
        print('You like a ...')
        pred = predict_breed_transfer(img_path)
        print(pred)

    else:
        print("I'm sorry, we weren't able to detect any dogs or humans in your picture!")
