import torch
import numpy as np
import torch.optim as optim
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms

from loguru import logger
import torch.nn as nn
import time
import json
from decimal import Decimal
import cv2
import sys
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
from scipy import fftpack
import math
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.cuda.set_device(3)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)  # 使用ResNet18作为基础模型
        self.fc = nn.Linear(1000, 2)  # 二分类任务，输出维度为2

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def train_CNN(params):
    All_Attack_Methods = params['Attack_Methods']
    # 定义路径
    path111 = "dataset/{}/train".format(All_Attack_Methods[0])
    path222 = "dataset/{}/train".format(All_Attack_Methods[0])

    if os.path.isdir(path111):
        file111 = get_image_list(path111)
    else:
        file111 = [path111]
    if os.path.isdir(path222):
        file222 = get_image_list(path222)
    else:
        file222 = [path222]
    logger.info(f"train_num={len(file111)},test_num={len(file222)}")
    time.sleep(10)

    logger.info(f"****开始训练*******")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # Initialize the dataset
    dataset = datasets.ImageFolder(root='dataset/{}/train'.format(All_Attack_Methods[0]), transform=transform)
    test_dataset = datasets.ImageFolder(root='dataset/{}/train'.format(All_Attack_Methods[0]), transform=transform)

    # Define batch size for DataLoader
    batch_size = 10

    # Initialize the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    logger.info(f"---training------")
    train_epochs = 20

    Best_Acc = 0
    for epoch in range(train_epochs):
        logger.info(f"---epoch={epoch}---")
        model.train()
        running_loss = 0
        train_correct = 0
        test_correct = 0
        total = 0
        total111 = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            predicted_train = torch.argmax(outputs.data, 1)
            train_correct += (predicted_train == labels).sum().item()
            # logger.info(f"predicted_train={predicted_train}={predicted_train.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.shape[0]
            running_loss += loss
        logger.info(
            f"epoch={epoch},Train_Accuarcy={train_correct}/{total}={train_correct / total},running_loss={running_loss}")
        with torch.no_grad():
            model.eval()
            for images111, labels111 in test_loader:
                images111, labels111 = images111.to(device), labels111.to(device)
                outputs111 = model(images111)
                predicted_test = torch.argmax(outputs111.data, 1)
                test_correct += (predicted_test == labels111).sum().item()
                total111 += images111.shape[0]
        Test_Accuarcy = test_correct / total111
        logger.info(f"epoch={epoch},Test_Accuarcy={test_correct}/{total111}={test_correct / total111}")
        if Test_Accuarcy > Best_Acc and Test_Accuarcy > 0.50:
            Best_Acc = Test_Accuarcy
            logger.info(f"成功保存！！！")
            torch.save(model.state_dict(),
                       'save_model/{}_CNN_Classifier_{}.pth'.format(All_Attack_Methods[0], Test_Accuarcy))

    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": All_Attack_Methods,
        'Save_path': 'save_model/{}_CNN_Classifier_{}.pth'.format(All_Attack_Methods[0], Test_Accuarcy)
    }

    return jsontext


def test_CNN(params):
    All_Attack_Methods = params['Attack_Methods']
    # 定义路径
    path111 = "dataset/{}/train".format(All_Attack_Methods[0])
    path222 = "dataset/{}/test".format(All_Attack_Methods[0])

    if os.path.isdir(path111):
        file111 = get_image_list(path111)
    else:
        file111 = [path111]
    if os.path.isdir(path222):
        file222 = get_image_list(path222)
    else:
        file222 = [path222]
    logger.info(f"train_num={len(file111)},test_num={len(file222)}")
    time.sleep(10)

    test_correct = 0
    total = 0
    total111 = 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # Initialize the dataset
    dataset = datasets.ImageFolder(root='dataset/{}/train'.format(All_Attack_Methods[0]), transform=transform)
    test_dataset = datasets.ImageFolder(root='dataset/{}/test'.format(All_Attack_Methods[0]), transform=transform)

    # Define batch size for DataLoader
    batch_size = 10

    # Initialize the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = CNN().to(device)
    if All_Attack_Methods[0] == 'FGSM':
        CNN_path = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection_keras_military/save_model/FGSM_CNN_Classifier_1.0.pth'
    elif All_Attack_Methods[0] == 'PSO_patch':
        CNN_path = '/data/Newdisk/chenjingwen/qkh/SJS_B4/yolo_zz_plus/save_model/PSO_patch_CNN_Classifier_0.65.pth'

    model.load_state_dict(torch.load(CNN_path))
    with torch.no_grad():
        model.eval()
        for images111, labels111 in test_loader:
            images111, labels111 = images111.to(device), labels111.to(device)
            outputs111 = model(images111)
            predicted_test = torch.argmax(outputs111.data, 1)
            test_correct += (predicted_test == labels111).sum().item()
            total111 += images111.shape[0]
    Test_Accuarcy = test_correct / total111
    logger.info(f"Test_Accuarcy={test_correct}/{total111}={test_correct / total111}")
    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": All_Attack_Methods,

    }

    return jsontext


def detect_main(params):
    if params['mode'] == 'train':
        result = train_CNN(params)
        return result
    elif params['mode'] == 'test':
        result = test_CNN(params)
        return result


if __name__ == "__main__":
    IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy"]
    params = {
        "taskId": 12345,
        "mode": 'test',  # {train、test}
        "Attack_Methods": ["FGSM"]  # {FGSM、IGSM、GuassianBlurAttack、PSO_patch
    }
    result = detect_main(params)
    print(result)
