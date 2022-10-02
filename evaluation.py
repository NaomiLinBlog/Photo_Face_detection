# -*- coding: utf-8 -*-                                                                                                            [48/99]
from cv2 import cv2 as cv2
import time
from MyDataset import MyDataset

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def main():
    # GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:', device)

    # Transform PIL to Tensor
    #資料預處理
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),#照片轉成張量
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
    
    # Data
    #trainingset = torchvision.datasets.ImageFolder(root='./LCC_FASD/training', transform=transform)
    trainingset = torchvision.datasets.ImageFolder(root='./LCC_FASD/development', transform=transform)
    print(trainingset.classes)
   

    # Data classes
    classes = trainingset.classes

    #evaluationset = torchvision.datasets.ImageFolder(root='./LCC_FASD/development', transform=transform)
    evaluationset = MyDataset(root='./LCC_FASD/evaluation', transform=transform, classes=classes)
    evaluationLoader = torch.utils.data.DataLoader(evaluationset, batch_size=8, shuffle=False, num_workers=2)                
    #讀取8張圖片, 不打亂,載入兩筆資料
    # Load pretrained_model
    model = torch.load('./LCC_FASD.pth', map_location=device)#讀取模型
    model.eval()#Freeze先前資料
    #model = nn.DataParallel(model)

    # Test
    checkpoint = 30
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():#不更新梯度
        since_epoch = time.time()
        for times, data in enumerate(evaluationLoader, 0):#times 從0開始計算index
        #for data in evaluationLoader:
            inputs, labels = data#data裡兩筆資料存入inputs,label[0]=real,label[1]=spoof
            # inputs = data[0]
            # labels = data[1]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #print(outputs)
            _, predicted = torch.max(outputs, 1)#取real,spoof兩行裡機率最大的那個決定real or spoof
            c = (predicted == labels).squeeze()#壓縮維度
            #labels   real spoof spoof real
            #preicted real real  spoof real
            #c        true false true  true
            for i in range(labels.size()[0]):#計算精確度
                label = labels[i]
                class_correct[label] += c[i].item() 
                class_total[label] += 1#統計true and faulse 各有幾個

            if (times+1) % checkpoint == 0 or times+1 == len(evaluationLoader):
                for i in range(len(classes)):
                    time_elapsed_epoch = time.time() - since_epoch
                    correct_percent = 0
                    if (class_total[i] > 0):
                        correct_percent = 100 * class_correct[i] / class_total[i]
                    print('[%d/%d] Accuracy of %5s : %2d %% in %.0fm %.0fs' % (
                        times+1, len(evaluationLoader), classes[i], correct_percent, time_elapsed_epoch // 60, time_elapsed_epoch % 60))

    correct = 0
    total = 0
    for i in range(len(classes)):
        correct += class_correct[i]
        total += class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    print('Accuracy of the network on the %d test inputs: %d %%' % (len(evaluationset), 100 * correct / total))

if __name__ == '__main__':
    main()
