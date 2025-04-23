#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/7/20 21:01
# @Author  : ZhangLongyuan
# @File    : Image_Attack.py
# @Software: PyCharm
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

# 聚类
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score

# SentimentDetect
import torch.nn.functional as F


def spectral_signature_scores(matrix_r):
    """
    :param matrix_r: Matrix of feature representations.
    :return: Outlier scores for each observation based on spectral signature.
    """
    matrix_m = matrix_r - np.mean(matrix_r, axis=0)
    # Following Algorithm #1 in paper, use SVD of centered features, not of covariance
    _, _, matrix_v = np.linalg.svd(matrix_m, full_matrices=False)
    eigs = matrix_v[:1]
    corrs = np.matmul(eigs, np.transpose(matrix_r))
    score = np.expand_dims(np.linalg.norm(corrs, axis=0), axis=1)
    return score


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy", ".PNG", '.JPEG']


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        # print(f"进入循环！！！")
        for filename in file_name_list:
            # print(f"进入下一层循环！！！")
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                # print(f"找到图像")
                image_names.append(apath)
    return image_names


def data_split(feature, labels):
    num_samples = feature.shape[0]
    partain = int(num_samples / 2)
    feature_pos, labels_pos = feature[:partain], labels[:partain]
    feature_neg, labels_neg = feature[partain:], labels[partain:]
    num_train = int(0.7 * partain)
    feature_train = np.concatenate((feature_pos[:int(num_train)], feature_neg[:num_train]))
    labels_train = np.concatenate((labels_pos[:int(num_train)], labels_neg[:num_train]))

    # feature_test = np.concatenate((feature_pos[num_train:], feature_neg[num_train:]))
    # labels_test= np.concatenate((labels_pos[num_train:], labels_neg[num_train:]))
    # a=random.uniform(1.4,1.41)
    feature_test = np.concatenate((feature_pos[int(num_train):], feature_neg[num_train:]))
    labels_test = np.concatenate((labels_pos[int(num_train):], labels_neg[num_train:]))
    # feature_test=feature_neg[num_train:]
    # labels_test=labels_neg[num_train:]

    return feature_train, feature_test, labels_train, labels_test


def data_split_test(feature, labels):
    num_samples = feature.shape[0]
    partain = int(num_samples / 2)
    feature_pos, labels_pos = feature[:partain], labels[:partain]
    feature_neg, labels_neg = feature[partain:], labels[partain:]
    num_train = int(0 * partain)
    feature_train = np.concatenate((feature_pos[:int(num_train)], feature_neg[:num_train]))
    labels_train = np.concatenate((labels_pos[:int(num_train)], labels_neg[:num_train]))

    # feature_test = np.concatenate((feature_pos[num_train:], feature_neg[num_train:]))
    # labels_test= np.concatenate((labels_pos[num_train:], labels_neg[num_train:]))
    # a=random.uniform(1.4,1.41)
    feature_test = np.concatenate((feature_pos[int(num_train):], feature_neg[num_train:]))
    labels_test = np.concatenate((labels_pos[int(num_train):], labels_neg[num_train:]))
    # feature_test=feature_neg[num_train:]
    # labels_test=labels_neg[num_train:]

    return feature_train, feature_test, labels_train, labels_test


def data_model_load_v2(Dataset, Model, device, batch_size):
    Dataset_dir = 'datasets'

    # 通用数据集
    if Dataset == 'imagenet_100':

        lable_txt = 'datasets/imagenet100_Chinese.xlsx'

        if Model == 'AlexNet':
            model = net.alexnet(pretrained=False)
            model.classifier[6] = nn.Linear(4096, 100)
            model = model.to(device)

        if Model == 'GoogleNet':
            model = net.googlenet(num_classes=100)
            # model.fc = nn.Linear(1024, 100)
            model = model.to(device)

        transform_imagenet = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, Dataset, 'train'),
            transform=transform_imagenet)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, Dataset, 'test'),
            transform=transform_imagenet)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=True, num_workers=0)

    # 军事数据集
    if Dataset == 'Military':

        lable_txt = 'datasets/Military_Chinese.xlsx'

        if Model == 'ResNet50':
            model = net.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 10)
            model = model.to(device)

        # zh添加
        if Model == 'VGG16':
            model = net.VGG16(num_classes=10)
            model = model.to(device)

        transform_imagenet = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, Dataset, 'train'),
            transform=transform_imagenet)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, Dataset, 'test'),
            transform=transform_imagenet)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=True, num_workers=0)

    return model, trainloader, testloader, testset, lable_txt


def data_model_load_v1(Dataset, Model, path, device):
    Dataset_dir = 'datasets'

    if Dataset == 'Military':

        lable_txt = 'datasets/Military_Chinese.xlsx'

        if Model == 'ResNet50':
            model = net_forward.resnet50(num_classes=10)
            # model.fc = nn.Linear(2048, 10)
            model = model.to(device)

        if Model == 'VGG16':
            model = net_forward.VGG16(num_classes=10)
            model = model.to(device)
            
        if Model == 'AlexNet':
            model = net_forward.AlexNet(num_classes=10)
            model = model.to(device)

        if Model == 'GoogleNet':
            model = net_forward.googlenet(num_classes=10)
            #model.fc = nn.Linear(1024, 10)
            model = model.to(device)

        transform_imagenet = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, Dataset, 'train'),
            transform=transform_imagenet)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=1,
                                                  shuffle=False, num_workers=0)
        testset = torchvision.datasets.ImageFolder(
            root=path,
            transform=transform_imagenet,
        )

        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=1,
                                                 shuffle=True, num_workers=0)

    if Dataset == 'imagenet_100':

        lable_txt = 'datasets/imagenet100_Chinese.xlsx'

        if Model == 'AlexNet':
            model = net_forward.alexnet()
            model.classifier[6] = nn.Linear(4096, 100)
            model = model.to(device)

        if Model == 'GoogleNet':
            model = net_forward.googlenet(num_classes=100)
            # model.fc = nn.Linear(1024, 100)
            model = model.to(device)

        transform_imagenet = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, Dataset, 'train'),
            transform=transform_imagenet)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=1,
                                                  shuffle=False, num_workers=0)
        testset = torchvision.datasets.ImageFolder(
            root=path,
            transform=transform_imagenet,
        )

        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=1,
                                                 shuffle=True, num_workers=0)

    if Dataset == 'warship':

        lable_txt = 'datasets/Military_Chinese.xlsx'

        if Model == 'ResNet50':
            model = net_forward.resnet50(num_classes=2)
            # model.fc = nn.Linear(2048, 10)
            model = model.to(device)

        if Model == 'VGG16':
            model = net_forward.VGG16(num_classes=2)
            model = model.to(device)

        if Model == 'AlexNet':
            model = net_forward.alexnet(num_classes=2)
            model = model.to(device)

        if Model == 'GoogleNet':
            model = net_forward.googlenet(num_classes=2)
            model = model.to(device)


        transform_imagenet = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, Dataset, 'train'),
            transform=transform_imagenet)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=1,
                                                  shuffle=False, num_workers=0)
        testset = torchvision.datasets.ImageFolder(
            root=path,
            transform=transform_imagenet,
        )

        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=1,
                                                 shuffle=True, num_workers=0)

    return model, testloader, testset


def npy_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    return np.load(path) / 255.0


def decimal2f(num):
    return float(int(num * 100) / 100)


def calc_norm(perturbation):
    try:
        c, h, w = perturbation.shape
    except:
        _, c, h, w = perturbation.shape
    L0 = np.sum(np.abs(np.sign(perturbation))) / (h * w * c)
    L2 = np.sqrt(np.sum(perturbation ** 2)) / (h * w * c)
    Li = np.max(np.abs(perturbation))
    return L0, L2, Li


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def arg_parse():
    parser = argparse.ArgumentParser(
        description='choose by yourself.')
    parser.add_argument('--Group',
                        type=str,
                        default='Image',
                        choices=['Image'],
                        help='type of Group')
    parser.add_argument('--Method',
                        type=str,
                        default='Attack',
                        choices=['Train', 'Attack'],
                        help='type of Method')

    parser.add_argument('--Dataset',
                        type=str,
                        default='MNIST',
                        help='type of dataset')
    parser.add_argument('--GPU',
                        type=int,
                        default=3,
                        choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='type of dataset')

    parser.add_argument('--Model',
                        type=str,
                        default='AlexNet',
                        help='model to be quantized')
    parser.add_argument('--Attack_Method',
                        type=str,
                        default='FGSM-Lambda1',
                        help='type of Attack_Method')

    parser.add_argument('--batch_size',
                        type=int,
                        default=5,
                        help='batch size of distilled data')
    parser.add_argument('--Attack', action='store_true', help="Attack",
                        default=True)

    args = parser.parse_args()
    return args


args = arg_parse()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,,6,7'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Current device:', torch.cuda.current_device(), args.GPU)


# SentimentDetect
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# SentimentDetect
def SD_train(classifier, detector, train_loader, criterion, optimizer, i, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    classifier.eval()
    detector.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        outputs, layer_out, layer_out_feature = classifier(data)
        feature = layer_out_feature
        optimizer.zero_grad()
        #        print(feature.shape)
        #        feature = feature.unsqueeze(0)
        #        feature = feature.reshape(27, 64, 16, 16)
        #        feature = feature.unsqueeze(0)
        #        print(feature.shape)

        output = detector(feature)

        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).float().sum(0)
        acc = correct.mul_(100.0 / data.size(0))

        losses.update(loss.item(), data.size(0))
        accs.update(acc.item(), data.size(0))

    print('Train Epoch: [{}/{}]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)'.format(
        i + 1, epoch, losses.avg, int(accs.sum / 100.0), accs.count, accs.avg))


# SentimentDetect
# def SD_evaluate(classifier, detector, val_loader,true_labels):
#     detect_result = []
#     accs = AverageMeter()
#     classifier.eval()
#     detector.eval()
#
#     with torch.no_grad():
#         for data, target in val_loader:
#
#             data, target = data.cuda(), target.cuda()
#             #            feature = classifier.feature_list(data)[1]
#             outputs, layer_out, layer_out_feature = classifier(data)
#             feature = layer_out_feature
#             output = detector(feature)
#
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#
#             for item in pred.cpu().numpy():
#
#                 detect_result.append(item[0])
#
#             correct = pred.eq(target.view_as(pred)).float().sum(0)
#             # print(correct)
#             acc = correct.mul_(100.0 / data.size(0))
#
#             accs.update(acc.item(), data.size(0))
#     print('\nEvaluate val set: Accuracy: {}/{} ({:.2f}%)\n'.format(
#         int(accs.sum / 100.0), accs.count, accs.avg))
#     return accs.avg, detect_result


def SD_evaluate(classifier, detector, val_loader):
    detect_result = []
    accs = AverageMeter()
    classifier.eval()
    detector.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            #            feature = classifier.feature_list(data)[1]
            outputs, layer_out, layer_out_feature = classifier(data)
            # print(outputs)
            feature = layer_out_feature
            # for i in range(len(feature)):
            # print(i,feature[i].shape)

            output = detector(feature)
            # print(output)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # print(output, '+++', pred.cpu().numpy())
            for item in pred.cpu().numpy():
                detect_result.append(item[0])
            correct = pred.eq(target.view_as(pred)).float().sum(0)
            acc = correct.mul_(100.0 / data.size(0))

            accs.update(acc.item(), data.size(0))
    print('\nEvaluate val set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(accs.sum / 100.0), accs.count, accs.avg))
    return accs.avg, detect_result


# Detector structure
class Text_CNN_AlexNet(torch.nn.Module):
    # Text-CNN Detector For ResNet-34
    def __init__(self):
        super(Text_CNN_AlexNet, self).__init__()
        self.cp1 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )

        filter_sizes = [1, 2, 3, 4]
        num_filters = 100
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (K, 512), padding=(0, 128), bias=True) for K in filter_sizes])
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

        out3 = self.cp3(x[-7])
        out3 = F.avg_pool2d(out3, 4)
        out3 = out3.view(out3.size(0), 1, -1)

        out4 = F.avg_pool2d(x[-2], 4)
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


class Text_CNN_GoogleNet(torch.nn.Module):
    # Text-CNN Detector For ResNet-34
    def __init__(self):
        super(Text_CNN_GoogleNet, self).__init__()
        self.cp1 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp2 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )

        filter_sizes = [1, 2, 3, 4]
        num_filters = 100
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (K, 512), bias=True) for K in filter_sizes])
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 200)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        out0 = self.cp1(x[0])
        out0 = self.cp2(out0)
        out0 = self.cp3(out0)
        out0 = F.avg_pool2d(out0, 4)
        out0 = out0.view(out0.size(0), 1, -1)

        out1 = self.cp1(x[0])
        out1 = self.cp2(out1)
        out1 = self.cp3(out1)
        out1 = F.avg_pool2d(out1, 4)
        out1 = out1.view(out1.size(0), 1, -1)

        out2 = self.cp2(x[3])
        out2 = self.cp3(out2)
        out2 = F.avg_pool2d(out2, 4)
        out2 = out2.view(out2.size(0), 1, -1)

        out3 = self.cp3(x[5])
        out3 = F.avg_pool2d(out3, 4)
        out3 = out3.view(out3.size(0), 1, -1)

        # 其余的
        # out4 = F.avg_pool2d(x[-2], 4)
        # out4 = out4.view(out4.size(0), 1, -1)
        # Military特制
        out4 = self.cp3(x[5])
        out4 = F.avg_pool2d(out4, 4)
        out4 = out4.view(out4.size(0), 1, -1)

        # print(x[0].shape,x[0].shape,x[3].shape,x[5].shape,x[-2].shape)
        # print(out0.shape, out1.shape, out2.shape, out3.shape, out4.shape)
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

# class Text_CNN_GoogleNet(torch.nn.Module):
#     # Text-CNN Detector For ResNet-34
#     def __init__(self):
#         super(Text_CNN_GoogleNet, self).__init__()
#         self.cp1 = nn.Sequential(
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
#             torch.nn.MaxPool2d((2, 2), stride=2)
#         )
#         self.cp2 = nn.Sequential(
#             nn.Conv2d(192, 256, kernel_size=3, padding=1),
#             torch.nn.MaxPool2d((2, 2), stride=2)
#         )
#         self.cp3 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             torch.nn.MaxPool2d((2, 2), stride=2)
#         )
#
#
#         filter_sizes = [1, 2, 3, 4]
#         num_filters = 100
#         self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (K, 512), bias=True) for K in filter_sizes])
#         self.dropout1 = nn.Dropout(0.1)
#         self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 200)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(200, 10)
#
#     def forward(self, x):
#         out0 = self.cp1(x[0])
#         out0 = self.cp2(out0)
#         out0 = self.cp3(out0)
#         out0 = F.avg_pool2d(out0, 4)
#         out0 = out0.view(out0.size(0), 1, -1)
#
#         out1 = self.cp1(x[0])
#         out1 = self.cp2(out1)
#         out1 = self.cp3(out1)
#         out1 = F.avg_pool2d(out1, 4)
#         out1 = out1.view(out1.size(0), 1, -1)
#
#         out2 = self.cp2(x[3])
#         out2 = self.cp3(out2)
#         out2 = F.avg_pool2d(out2, 4)
#         out2 = out2.view(out2.size(0), 1, -1)
#
#         out3 = self.cp3(x[5])
#         out3 = F.avg_pool2d(out3, 4)
#         out3 = out3.view(out3.size(0), 1, -1)
#
#         out4 = F.avg_pool2d(x[-2], 4)
#         out4 = out4.view(out4.size(0), 1, -1)
#
#         txt = torch.cat((out0, out1, out2, out3, out4), 1)
#         txt = torch.unsqueeze(txt, 1)
#         #        out = [F.relu(conv(txt)).squeeze(3) for conv in self.convs]
#         out = [F.relu(conv(txt)).sum(2) / F.relu(conv(txt)).size(2) for conv in self.convs]
#         out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
#         out = torch.cat(out, 1)
#         out = self.dropout1(out)
#         out = F.relu(self.fc1(out))
#         out = self.dropout2(out)
#         logit = self.fc2(out)
#
#         return logit


class Text_CNN_VGG(torch.nn.Module):
    # Text-CNN Detector For ResNet-34
    def __init__(self):
        super(Text_CNN_VGG, self).__init__()
        self.cp1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
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

        out2 = self.cp2(x[5])
        out2 = self.cp3(out2)
        out2 = F.avg_pool2d(out2, 4)
        out2 = out2.view(out2.size(0), 1, -1)

        out3 = self.cp3(x[14])
        out3 = F.avg_pool2d(out3, 4)
        out3 = out3.view(out3.size(0), 1, -1)

        out4 = F.avg_pool2d(x[5], 8)
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


def SentimentDetect(params1):
    Dataset = params1['Dataset']
    Model = params1['Model']
    Attack_Name = params1['Attack_Name']

    print("Dataset:{} Model:{} Attack_Method:{}".format(Dataset, Model, Attack_Name))

    image_clean = []
    ture_label = []
    image_adv = []
    ture_label_adv = []
    features = []
    features_adv = []
    # 读取500/1000张干净样本，数量由pickNumber_v1控制
    # path = os.path.join('datasets', Dataset, 'test')

    ori_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/{}/{}/select_data/{}/png/ori'.format(Dataset, Model, Attack_Name)
    #ori_path = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/{}/{}/{}/ori'.format(Dataset, Model, Attack_Name)

    model, testloader, testset = data_model_load_v1(Dataset, Model, ori_path, device)
    model_path = 'model_data/{}_{}.pth'.format(Model, Dataset)

    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    pickNumber_v1 = 400  # 控制训练二分类器所用原始样本数量
    file111 = get_image_list(ori_path)
    testloader = random.sample(list(testloader), pickNumber_v1)

    for i, (image, label) in enumerate(testloader):
        model.eval()
        image, label = image.to(device), label.to(device)

        label = np.array(label.detach().cpu())
        image = np.array(image.detach().cpu())

        ture_label.append(label)
        image_clean.append(image)

    # 读取500/1000张对抗样本，数量由pickNumber_v2控制
    adv_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/{}/{}/select_data/{}/png/adv'.format(Dataset, Model, Attack_Name)
    #adv_path = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/{}/{}/{}/adv'.format(Dataset, Model, Attack_Name)

    _, testloader_adv, testset_adv = data_model_load_v1(Dataset, Model, adv_path, device)
    pickNumber_v2 = 400
    testloader_adv = random.sample(list(testloader_adv), pickNumber_v2)
    for i, (adv_image, adv_label) in enumerate(testloader_adv):
        model.eval()
        adv_image, adv_label = adv_image.to(device), adv_label.to(device)
        # outputs, layer_out, layer_out_feature = model(adv_image)
        # layer_out_feature[-1].requires_grad_(True)
        # outputs.requires_grad_(True)
        # pred = torch.argmax(outputs)
        # pred_label = outputs[:, pred]
        # temp_feature = layer_out_feature[-2]
        # if Model == 'AlexNet':
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 256 * 8 * 8)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # elif Model == "VGG16":
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 512 * 7 * 7)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # elif Model == "ResNet50":
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 2048 * 7 * 7)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # elif Model == "GoogleNet":
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 1024 * 9 * 9)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # else:
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        #     temp_feature = torch.squeeze(temp_feature, dim=1)
        #     temp_feature = torch.squeeze(temp_feature, dim=1)
        #
        # temp_feature = np.array(temp_feature.detach().cpu())
        # features_adv.append(temp_feature)

        adv_label = np.array(adv_label.detach().cpu())
        adv_image = np.array(adv_image.detach().cpu())

        ture_label_adv.append(adv_label)
        image_adv.append(adv_image)

        # get_loader()
    image_clean = np.array(image_clean)
    image_adv = np.array(image_adv)
    inputs = np.vstack((image_clean, image_adv))

    features = np.array(features)
    features_adv = np.array(features_adv)
    features = np.vstack((features, features_adv))

    ture_label = np.array(ture_label)
    ture_label_adv = np.array(ture_label_adv)

    labels_clean1 = np.ones((ture_label.shape[0], 1))
    labels_adv1 = np.zeros((ture_label_adv.shape[0], 1))
    labels = np.vstack((labels_clean1, labels_adv1))

    true_labels = np.vstack((ture_label, ture_label_adv))

    x_train, x_test, y_train, y_test = data_split_test(inputs, labels)
    x_train, x_test, y_train, y_test = torch.tensor(x_train).to(device), torch.tensor(x_test).to(
        device), torch.FloatTensor(y_train).to(device), torch.FloatTensor(y_test).to(device)

    x_train = x_train.squeeze(1)
    y_train = y_train.squeeze(1)
    #    print('x_train',x_train)
    #    print(x_train.shape)
    #    print(y_train)
    #    print(y_train.shape)
    x_test = x_test.squeeze(1)
    y_test = y_test.squeeze(1)

    # train_dataset = Data.TensorDataset(x_train, y_train)
    test_dataset = Data.TensorDataset(x_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=27, shuffle=True)

    if Model == 'AlexNet':
        detector = Text_CNN_AlexNet().cuda()
    if Model == 'GoogleNet':
        detector = Text_CNN_GoogleNet().cuda()
    if Model == 'ResNet50':
        detector = Text_CNN_ResNet().cuda()
    if Model == 'VGG16':
        detector = Text_CNN_VGG()
    detector.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=0.0001)
    epoch = 10
    max_acc = 0

    # print('Training Detector...')
    # for i in range(epoch):
    #     SD_train(model, detector, train_dataloader, criterion, optimizer, i, epoch)
    #     acc = SD_evaluate(model, detector, test_dataloader)
    #     if acc > max_acc:
    #         print('saveing new best model ckpt for epoch #{}'.format(i + 1))
    #         torch.save(detector.state_dict(), "model_data/detect/SentimentDetect_%s_%s_%s.pth" % (Dataset, Model, Attack_Name))
    #         max_acc = acc

    # Evaluating Detector
    print('Evaluating Detector...')
    # detector.load_state_dict(
    #     torch.load("model_data/detect/SentimentDetect_%s_%s_%s.pth" % (Dataset, Model, Attack_Name),
    #                map_location='cpu'))

    detector.load_state_dict(
        torch.load(
            "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/model_data/detect2/SentimentDetect_warship_AlexNet_All-Attack-v2.pth",
            map_location='cpu'))
    Test_Accuarcy, detect_result = SD_evaluate(model, detector, test_dataloader)
    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": Attack_Name,
        "detect_result": detect_result
    }

    return jsontext


def SentimentTrain(params1):
    Dataset = params1['Dataset']
    Model = params1['Model']
    Attack_Name = params1['Attack_Name']

    print("Dataset:{} Model:{} Attack_Method:{}".format(Dataset, Model, Attack_Name))

    image_clean = []
    ture_label = []
    image_adv = []
    ture_label_adv = []
    features = []
    features_adv = []
    # 读取500/1000张干净样本，数量由pickNumber_v1控制
    path = os.path.join('datasets', Dataset, 'train')
    model, testloader, testset = data_model_load_v1(Dataset, Model, path, device)
    model_path = 'model_data/{}_{}.pth'.format(Model, Dataset)

    # model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.load_state_dict(torch.load(model_path, map_location={'cuda:3': 'cuda:2'}))
    pickNumber_v1 = 3900  # 控制训练二分类器所用原始样本数量
    file111 = get_image_list(path)
    print(len(file111))
    #    logger.info(f"file111={len(file111)}")
    #    logger.info(f"testloader={len(testloader)}")
    #    logger.info(f"testset={len(testset)}")
    testloader = random.sample(list(testloader), pickNumber_v1)
    #    print("ori_nums=", len(testloader))
    for i, (image, label) in enumerate(testloader):
        model.eval()
        image, label = image, label
        # outputs, layer_out, layer_out_feature = model(image)
        # layer_out_feature[-1].requires_grad_(True)
        # outputs.requires_grad_(True)
        # pred = torch.argmax(outputs)
        # pred_label = outputs[:, pred]
        #
        # temp_feature = layer_out_feature[-2]
        # if Model == 'AlexNet':
        #     #            temp_feature = temp_feature.reshape(temp_feature.size(0), 256 * 6 * 6)
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 256 * 8 * 8)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # elif Model == "VGG16":
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 512 * 7 * 7)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # elif Model == "ResNet50":
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 2048 * 7 * 7)
        #     # DBA
        #     #            temp_feature = temp_feature.reshape(temp_feature.size(0), 2048 * 1 * 1)
        #
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # elif Model == "GoogleNet":
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 1024 * 9 * 9)
        #     # DBA
        #     #            temp_feature = temp_feature.reshape(temp_feature.size(0), 1024 * 1 * 1)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # else:
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        #     temp_feature = torch.squeeze(temp_feature, dim=1)
        #     temp_feature = torch.squeeze(temp_feature, dim=1)
        #
        # temp_feature = np.array(temp_feature.detach().cpu())
        # features.append(temp_feature)

        label = np.array(label.detach().cpu())
        image = np.array(image.detach().cpu())

        ture_label.append(label)
        image_clean.append(image)

    # 读取500/1000张对抗样本，数量由pickNumber_v2控制
    
    # adv_path = 'datasets/gen_data/{}/{}/{}/png/adv'.format(Dataset, Model, Attack_Name)
    adv_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/warship/ResNet50/All-Attack'

    _, testloader_adv, testset_adv = data_model_load_v1(Dataset, Model, adv_path, device)
    print(len(testset_adv))
    pickNumber_v2 = 3900
    testloader_adv = random.sample(list(testloader_adv), pickNumber_v2)
    for i, (adv_image, adv_label) in enumerate(testloader_adv):
        model.eval()
        adv_image, adv_label = adv_image, adv_label
        # outputs, layer_out, layer_out_feature = model(adv_image)
        # layer_out_feature[-1].requires_grad_(True)
        # outputs.requires_grad_(True)
        # pred = torch.argmax(outputs)
        # pred_label = outputs[:, pred]
        # temp_feature = layer_out_feature[-2]
        # if Model == 'AlexNet':
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 256 * 8 * 8)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # elif Model == "VGG16":
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 512 * 7 * 7)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # elif Model == "ResNet50":
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 2048 * 7 * 7)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # elif Model == "GoogleNet":
        #     temp_feature = temp_feature.reshape(temp_feature.size(0), 1024 * 9 * 9)
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        # else:
        #     temp_feature = torch.squeeze(temp_feature, dim=0)
        #     temp_feature = torch.squeeze(temp_feature, dim=1)
        #     temp_feature = torch.squeeze(temp_feature, dim=1)
        #
        # temp_feature = np.array(temp_feature.detach().cpu())
        # features_adv.append(temp_feature)

        adv_label = np.array(adv_label.detach().cpu())
        adv_image = np.array(adv_image.detach().cpu())

        ture_label_adv.append(adv_label)
        image_adv.append(adv_image)

        # get_loader()
    image_clean = np.array(image_clean)
    image_adv = np.array(image_adv)
    inputs = np.vstack((image_clean, image_adv))

    features = np.array(features)
    features_adv = np.array(features_adv)
    features = np.vstack((features, features_adv))

    ture_label = np.array(ture_label)
    ture_label_adv = np.array(ture_label_adv)

    labels_clean1 = np.ones((ture_label.shape[0], 1))
    labels_adv1 = np.zeros((ture_label_adv.shape[0], 1))
    labels = np.vstack((labels_clean1, labels_adv1))

    x_train, x_test, y_train, y_test = data_split(inputs, labels)
    x_train, x_test, y_train, y_test = torch.tensor(x_train).to(device), torch.tensor(x_test).to(
        device), torch.FloatTensor(y_train).to(device), torch.FloatTensor(y_test).to(device)

    x_train = x_train.squeeze(1)
    y_train = y_train.squeeze(1)
    #    print('x_train',x_train)
    #    print(x_train.shape)
    #    print(y_train)
    #    print(y_train.shape)
    x_test = x_test.squeeze(1)
    y_test = y_test.squeeze(1)

    train_dataset = Data.TensorDataset(x_train, y_train)
    test_dataset = Data.TensorDataset(x_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=3, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

    if Model == 'AlexNet':
        detector = Text_CNN_AlexNet()
    if Model == 'GoogleNet':
        detector = Text_CNN_GoogleNet()
    if Model == 'ResNet50':
        detector = Text_CNN_ResNet()
    if Model == 'VGG16':
        detector = Text_CNN_VGG()
    detector.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=0.0001)
    epoch = 50
    max_acc = 0

    print('Training Detector...')
    for i in range(epoch):
        SD_train(model, detector, train_dataloader, criterion, optimizer, i, epoch)
        acc, _ = SD_evaluate(model, detector, test_dataloader)
        if acc > max_acc:
            print('saveing new best model ckpt for epoch #{}'.format(i + 1))
            torch.save(detector.state_dict(),
                       "model_data/detect/SentimentDetect_%s_%s_%s.pth" % (Dataset, Model, Attack_Name))
            max_acc = acc
    jsontext = {
        'Accuracy': max_acc,
        "Attack_Method": Attack_Name,
        "save_path": "model_data/detect/SentimentDetect_%s_%s_%s.pth" % (Dataset, Model, Attack_Name)
    }

    return jsontext


def detect_main(params):
    if params['mode'] == 'train':
        result = SentimentTrain(params)
        return result
    elif params['mode'] == 'test':
        result = SentimentDetect(params)
        return result


if __name__ == "__main__":

    # AttackMethodList = ['FGSM-Lambda1','DeepFool-Lambda1','FGSM-L1','MIFGSM-Lambda1','EAD-Lambda1','PGD-Lambda1','PatchAttack',
    # 'JSMA-Lambda1','Adef-Lambda1','NFA-Lambda1','OnePixel','PointwiseAttack','GaussianBlurAttack','ZOO-Lambda1','Boundary']
    # AttackMethodList = ['FGSM-Lambda1', 'DeepFool-Lambda1', 'FGSM-L1', 'MIFGSM-Lambda1', 'PatchAttack', 'JSMA-Lambda1',
                        # 'PGD-Lambda1', 'OnePixel', 'EAD-Lambda1', 'GaussianBlurAttack', 'ZOO-Lambda1', 'Boundary']
    AttackMethodList = ['OnePixel']
    # AttackMethodList = ['FGSM-L1','MIFGSM-Lambda1','PGD-Lambda1','OnePixel','GaussianBlurAttack','ZOO-Lambda1','GaussianBlurAttack']

    for AttackMethod in AttackMethodList:
        params = {
            'Dataset': 'warship',
            'mode': 'test',
            'Model': 'AlexNet',
            'Attack_Name': AttackMethod
        }
        result = detect_main(params)
        print(result)
