import sys

import torch
import numpy as np
import torch.optim as optim
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision import datasets, transforms
import net
from imageio import imwrite, imread
import matplotlib.pyplot as plt
import argparse
from utils.tool import *
from utils.D_D_tool import *
import time
import json
from decimal import Decimal
import cv2
import sporco.metric as sm
from sporco.admm import cbpdn
from sporco import util
import net_forward
from utils import net_Det
import torch.utils.data as Data
# 后门检测 czq
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score
from loguru import logger
import random
from PIL import Image
# 聚类
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score

# SentimentDetect
import torch.nn.functional as F

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


device = torch.device("cpu")


class Text_CNN_ResNet(torch.nn.Module):
    # Text-CNN Detector For ResNet-34
    def __init__(self):
        super(Text_CNN_ResNet, self).__init__()
        self.cp1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )

        filter_sizes = [1, 2, 3, 4]
        num_filters = 100
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (K, 512), bias=True) for K in filter_sizes])
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 200)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        out0 = self.cp1(x[0])
        out0 = self.cp2(out0)
        out0 = self.cp3(out0)
        out0 = F.avg_pool2d(out0, 4)
        out0 = out0.view(out0.size(0), 1, -1)

        out1 = self.cp1(x[1])
        out1 = self.cp2(out1)
        out1 = self.cp3(out1)
        out1 = F.avg_pool2d(out1, 4)
        out1 = out1.view(out1.size(0), 1, -1)

        out2 = self.cp2(x[3])
        out2 = self.cp3(out2)
        out2 = F.avg_pool2d(out2, 4)
        out2 = out2.view(out2.size(0), 1, -1)

        out3 = self.cp3(x[4])
        out3 = F.avg_pool2d(out3, 7)
        out3 = F.interpolate(out3, size=(3, 3))
        out3 = out3.view(out3.size(0), 1, -1)

        out4 = F.avg_pool2d(x[7], 4)
        out4 = F.interpolate(out4, size=(3, 3))
        out4 = out4.view(out4.size(0), 1, -1)

        txt = torch.cat((out0, out1, out2, out3, out4), 1)
        txt = torch.unsqueeze(txt, 1)
        #        out = [F.relu(conv(txt)).squeeze(3) for conv in self.convs]
        out = [F.relu(conv(txt)).sum(2) / F.relu(conv(txt)).size(2) for conv in self.convs]
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        logit = self.fc2(out)

        return logit


def model_load(model_path, type):
    if (type == 'classify'):
        model = net_forward.resnet50(num_classes=10)

    if (type == 'detect'):
        model = Text_CNN_ResNet()

    model = model.to(device)
#     model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    return model


def resnet_fgsm_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()

def resnet_patchattack_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()



def resnet_deepfool_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()

def resnet_igsm_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()

def resnet_mifgsm_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()

def resnet_pgd_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()

def resnet_finefool_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()


def resnet_jsma_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()



def resnet_onepixel_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()

def resnet_uap_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()

def resnet_zoo_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()

def resnet_boundary_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    classification_model.eval()
    detection_model.eval()
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transformed_image = transform(img).unsqueeze(0).to(device)
    img.close()

    outputs, layer_out, layer_out_feature = classification_model(transformed_image)

    feature = layer_out_feature
    # for i in range(len(feature)):
    # print(i,feature[i].shape)

    output = detection_model(feature)
    # print(output)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    return pred.item()


if __name__ == "__main__":
    FGSM_IMAGES_PATH = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/ResNet50/select_data/FGSM-Lambda1/png"  # 测试数据集路径

    RESNET50_MODEL_PATH = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/model_data/ResNet50_Military.pth'
    FGSM_MODEL_PATH = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/model_data/detect/SentimentDetect_Military_ResNet50_All-Attack.pth"  # 测试检测模型路径

    model = []
    FGSM_MODEL = model_load(FGSM_MODEL_PATH, "detect")
    RESNET50_MODEL = model_load(RESNET50_MODEL_PATH, "classify")
    model.append(FGSM_MODEL)
    model.append(RESNET50_MODEL)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    results = []
    labels = []
    total_time = 0
    for root, dirs, files in os.walk(FGSM_IMAGES_PATH):  # 以该数据集为例，可替换

        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)

                if 'ori' in file_path.split('/'):
                    labels.append(1)
                else:
                    labels.append(0)
                with open(file_path, "rb") as img_file:
                    start_time = time.time()
                    result = resnet_fgsm_detection(img_file, model)
                    end_time = time.time()
                    results.append(result)
                    total_time += (end_time - start_time)
    correct_predictions = sum(p == l for p, l in zip(results, labels))
    accuracy = correct_predictions / len(labels)
    avg_time = total_time / len(labels)
    print("ACC: " + str(accuracy))
    print("avg_time: " + str(avg_time))
    # print(results)
    # print(labels)
