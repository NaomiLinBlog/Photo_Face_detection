# -*- coding: utf-8 -*-        
import os
import time
import random

import numpy
from cv2 import cv2 as cv2
from PIL import Image
from PIL import ImageDraw
import face_recognition

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

path = "LCC_FASD/evaluation"
#path = "LCC_FASD/development/spoof"


def prepare():
    full_path = os.path.realpath(__file__)
    print(full_path + "\n")
    root_path, _ = os.path.split(full_path)
    os.chdir(root_path)

    # GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:', device)

    # Transform PIL to Tensor
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    # Data
    #trainingset = torchvision.datasets.ImageFolder(root='./LCC_FASD/training', transform=transform)
    trainingset = torchvision.datasets.ImageFolder(root='./LCC_FASD/development', transform=transform)
    print(trainingset.classes)

    # Data classes
    classes = trainingset.classes

    # Load pretrained_model
    model = torch.load('./LCC_FASD.pth', map_location=device)
    model.eval()
    #model = nn.DataParallel(model)

    return model, transform, classes

def evaluation(image, model, transform, classes, device=None):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
    with torch.no_grad():
        since_epoch = time.time()
        inputs = transform(image).unsqueeze(0)
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.squeeze().item()

        time_elapsed_epoch = time.time() - since_epoch
        print('Evaluation this image is %5s in %.3fs' % (classes[predicted], time_elapsed_epoch % 60))
        return classes[predicted], predicted

def image_show(pil_image):
    img = cv2.cvtColor(numpy.asarray(pil_image),cv2.COLOR_RGB2BGR)
    #img = numpy.asarray(pil_image)

    cv2.imshow('tmp', img)
    c = cv2.waitKeyEx(0) % 256

    if c == ord('q') or c ==  27:
        exit()
    elif cv2.getWindowProperty('tmp', 1) == -1:
        exit()
    else:
        print('you press ' + str(c))

def main():
    model, transform, classes = prepare()

    files = os.listdir(path)
    random.shuffle(files)
    
    for file_name in files:
        full_path = path + "/" + file_name
        label = file_name.split('_')[0]
        print(full_path)

        #image = face_recognition.load_image_file(full_path)
        image = cv2.imread(full_path)
        #print(img.shape)
        
        pil_image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).convert('RGB')
        #pil_image = Image.fromarray(image).convert('RGB')
        result, _ = evaluation(pil_image, model, transform, classes)

        resized_pil_image = pil_image.resize((300, 300), Image.BILINEAR)
        resized_image = cv2.cvtColor(numpy.asarray(resized_pil_image),cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(resized_image)

        for (top, right, bottom, left) in face_locations:
            # Create a Pillow ImageDraw Draw instance to draw with
            draw = ImageDraw.Draw(resized_pil_image)

            fill_color = (255, 0, 0) # red
            if label == result:
                fill_color = (0, 0, 255) # blue
            
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=fill_color)

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(result)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=fill_color, outline=fill_color)
            draw.text((left + 6, bottom - text_height - 5), result, fill=(255, 255, 255, 255))

            image_show(resized_pil_image)

if __name__ == '__main__':
    main()

    