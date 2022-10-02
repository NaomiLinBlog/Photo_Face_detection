# -*- coding: utf-8 -*-
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pretrainedmodels

def main():
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
    trainingset = torchvision.datasets.ImageFolder(root='./LCC_FASD/training', transform=transform)
    #trainingset = torchvision.datasets.ImageFolder(root='./LCC_FASD/development', transform=transform)
    trainingLoader = torch.utils.data.DataLoader(trainingset, batch_size=8, shuffle=True, num_workers=2)
    print(trainingset.classes)#自動分類
    print(trainingset.class_to_idx)
    
    # Data classes
    classes = trainingset.classes#object中提取member(real,spoof)給classes

    model_name = "senet154"
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    #pretrainedmodels.senet154()可更換model
    #model.avg_pool = nn.AvgPool2d(int(image_size / 32), stride=1)
    model.last_linear = nn.Linear(model.last_linear.in_features, len(classes)) #分類分類再分類成兩類real or spoof 1000->2
    model = model.to(device)
    #print(model)

    # Parameters
    criterion = nn.CrossEntropyLoss()
    lr = 0.0001 #learning rate,slow 比較不會出現擬合過度的情況
    epochs = 30 #學習30次
    optimizer = optim.Adam(model.parameters(), lr=lr)#優化器Adam
    checkpoint = 10


    start_time = time.time()
    # Train
    for epoch in range(epochs):
        running_loss = 0.0
        since_epoch = time.time()

        for times, data in enumerate(trainingLoader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients 
            optimizer.zero_grad() 
            #一         二         三
            # forward + backward + optimize about learning rate
            outputs = model(inputs)
            loss = criterion(outputs, labels)#與目標差多少距離
            loss.backward()
            optimizer.step()#更新

            # print statistics
            running_loss += loss.item()#總共學習了多少(越小越接近目標學不到東西)

            if (times+1) % checkpoint == 0 or times+1 == len(trainingLoader):
                time_elapsed_epoch = time.time() - since_epoch
                print('[%d/%d, %d/%d] loss: %.3f in %.0fm %.0fs' % (
                    epoch+1, epochs, times+1, len(trainingLoader), running_loss/2000, time_elapsed_epoch // 60, time_elapsed_epoch % 60))

    time_elapsed = time.time() - start_time
    print('Finished Training in %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))
    torch.save(model, 'LCC_FASD.pth')

if __name__ == '__main__':
    main()