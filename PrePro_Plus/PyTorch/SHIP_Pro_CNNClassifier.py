import torch
import numpy as np
import torch.optim as optim
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from art.defences.preprocessor import SpatialSmoothing
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
import torch.nn as nn
import argparse
from loguru import logger
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
import time
import torchvision.transforms.functional as TF
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
import torchvision.models as models
# from ssd_v1 import SSD
from yolo import YOLO
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1")

import subprocess


def get_gpu_utilization():
    # 运行 nvidia-smi 命令并获取输出
    smi_output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
    ).decode('utf-8').strip()

    # 将输出转换为整数列表，每个 GPU 的利用率为列表的一个元素
    gpu_utilization = [int(x) for x in smi_output.split('\n')]
    return gpu_utilization


def make_parser():
    parser = argparse.ArgumentParser(description='Statistical Detection on YOLOX')
    parser.add_argument('--gpu', '--gpu_ids', default='2', type=str)
    parser.add_argument('--phase', default='train2017', type=str)
    parser.add_argument('-b', '--batch_size', default=256, type=int)
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default='yolox-x', help="model name")
    parser.add_argument("--path", default="", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", default=True, action="store_true",
                        help="whether to save the inference result of image/video", )
    parser.add_argument("-f", "--exp_file", default=None, type=str,
                        help="please input your experiment description files", )
    parser.add_argument("-c", "--ckpt", default='', type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision evaluating.", )
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true",
                        help="To be compatible with older versions", )
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true",
                        help="Fuse conv and bn for testing.", )
    parser.add_argument("--trt", dest="trt", default=False, action="store_true",
                        help="Using TensorRT model for testing.", )
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu", )
    parser.add_argument('--DetectSet',
                        type=str,
                        default=['statistical_detection'],
                        choices=['statistical_detection', 'Total_VarMin', 'Compression', 'Feature_Squeezing'],
                        help='type of Detect_Method')
    return parser


class linear_model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(linear_model, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 512)
        self.act1 = torch.nn.ReLU(True)
        self.linear2 = torch.nn.Linear(512, 512)
        self.act2 = torch.nn.ReLU(True)
        self.linear3 = torch.nn.Linear(512, 512)
        self.act3 = torch.nn.ReLU(True)
        self.linear4 = torch.nn.Linear(512, output_size)
        #        self.act4 = torch.nn.ReLU(True)
        #        self.linear5 = torch.nn.Linear(1024, output_size)
        self.act5 = torch.nn.Softmax(dim=1)
        # self.act5 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.act3(x)
        x = self.linear4(x)
        # x = self.act4(x)
        # x = self.linear5(x)
        x = self.act5(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(85 * 8400, 1029)  # 修改全连接层的输入尺寸
        self.fc2 = nn.Linear(1029, 2)
        # self.fc1 = nn.Linear(85, 10290)  # 修改全连接层的输入尺寸
        # self.fc2 = nn.Linear(10290, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=8)  # 批归一化层
        self.relu = nn.ReLU()

        # 定义池化层
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # # # 添加第二个卷积层
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm1d(num_features=32)        
        # self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc1 = nn.Linear(8 * 357000, 256)  # 根据输入形状计算得到
        self.dropout1 = nn.Dropout(p=0.5)  # 添加Dropout层
        self.fc2 = nn.Linear(256, 2)  # 二分类任务，输出两个类别

    def forward(self, x):
        # 将输入张量转换成合适的形状
        x = x.view(-1, 1, 714000)

        # 执行卷积、激活函数和池化操作
        x = self.conv1(x)
        x = self.bn1(x)  # 应用批归一化
        x = self.relu(x)
        x = self.maxpool(x)

        # # 添加第二个卷积层和批归一化层
        # x = self.conv2(x)
        # x = self.bn2(x)

        # 将张量展平以输入全连接层
        x = x.view(-1, 8 * 357000)  # 根据池化层的输出形状计算得到
        x = self.fc1(x)
        x = self.dropout1(x)  # 应用Dropout层
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出维度为2，用于二分类任务
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class binary_model(nn.Module):
    def __init__(self, num_classes):
        super(binary_model, self).__init__()
        self.model0 = Sequential(
            # 0
            # 输入3通道、输出64通道、卷积核大小、步长、补零、
            Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2, padding=3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool2d((1, 1))
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        self.fc = Linear(512, num_classes)

    def forward(self, x):
        x = self.model0(x)

        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)

        # 最后3个
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy"]


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


# 获取检测框的中心点
def get_center(box):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


# 计算框之间的距离
def box_distance(box1, box2):
    center1 = get_center(box1)
    center2 = get_center(box2)
    # logger.info(f"center1={center1}")
    # logger.info(f"center2={center2}")
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    return distance


# 找到最大的一个框
def max_boxDistance(box):
    x1, y1, x2, y2 = box
    distance = int(math.fabs((x2 - x1) * (y2 - y1)))
    return distance


# 图片裁剪和保存
def crop_and_save_image(image_path, box, save_path):
    image = Image.open(image_path)

    logger.info(f"box111={box}")
    box[0] = max(box[0], 0)
    box[1] = max(box[1], 0)
    box[2] = min(box[2], image.size[1])
    box[3] = min(box[3], image.size[0])
    logger.info(f"image={image.size}")
    logger.info(f"box222={box}")
    logger.info(f"image_path={image_path}")
    x1, y1, x2, y2 = box.tolist()  # top,left,bottom,right
    a1, a2, b1, b2 = math.ceil(x1), math.ceil(y1), int(math.fabs(x2 - x1)), int(math.fabs(y2 - y1))

    cropped_image = image.crop([y1, x1, y2, x2])

    cropped_image.save(save_path)


def train_binary_classifier(test_image_dir, Attack_Method, ori_path, yolos):
    # 建立目录
    # attack_methods=['FGSM','IGSM','MIFGSM','PGD','patch_attack','GuassianBlurAttack','backdoor']
    attack_methods = ['ALL_warship_v2']
    for i in range(len(attack_methods)):
        os.makedirs(test_image_dir + "/{}/train/adv".format(attack_methods[i]), exist_ok=True)
        os.makedirs(test_image_dir + "/{}/test/adv".format(attack_methods[i]), exist_ok=True)
        os.makedirs(test_image_dir + "/{}/train/ori".format(attack_methods[i]), exist_ok=True)
        os.makedirs(test_image_dir + "/{}/test/ori".format(attack_methods[i]), exist_ok=True)
    save = True

    image_files = []
    oriImage_number = 0
    ground_label = []
    # 训练集
    train_path = []
    train_ground_label = []
    # 验证集
    test_path = []
    test_ground_label = []
    # # 检测集

    # pickNumber_v1=2400 #2500 白盒攻击正常样本2000张，'FGSM','IGSM','MIFGSM','PGD'','patch_attack'各500张
    # pickNumber_v1 = 500 #黑盒攻击正常样本500张，GaussianBlurAttack也500张

    random.seed(7)
    ###读取对抗样本
    # All_Attack_Methods=['FGSM','IGSM','MIFGSM','PGD','patch_attack','GuassianBlurAttack','backdoor']
    All_Attack_Methods = [Attack_Method]
    # All_Attack_Methods = ['GuassianBlurAttack']
    # All_Attack_Methods = ['backdoor']
    # All_Attack_Methods=['mosaic']

    for i in range(len(All_Attack_Methods)):
        advImage_number = 0
        if All_Attack_Methods[i] == 'backdoor':
            adv_path = '/data1/BigPlatform/ZJPlatform/bice_pingce/002-Code/001_ObjectDetection/002_Code/extra_task/yolox_new/backdoor/data/jpg/'
        elif All_Attack_Methods[i] == 'ALL_warship':
            adv_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/all_attack/adv'
        else:
            adv_path = "/data1/BigPlatform/ZJPlatform/bice_pingce/002-Code/001_ObjectDetection/002_Code/extra_task/yolox_new/yolox_ship_adv_generation/adv_data/{}".format(
                All_Attack_Methods[i])

        # pickNumber_v2=pickNumber_v1
        if os.path.isdir(adv_path):
            adv_file = get_image_list(adv_path)
        else:
            adv_file = [adv_path]

        adv_sample_number = len(adv_file) / len(All_Attack_Methods)
        logger.info(f"adv_sample_number={adv_sample_number}")
        pickNumber_adv = int((adv_sample_number / 4) * 3)
        logger.info(f"pickNumber_adv={pickNumber_adv}")
        number1 = 0
        number2 = 0
        random.shuffle(adv_file)
        for image_name in adv_file:
            image_files.append(image_name)
            ground_label.append(1)
            advImage_number += 1
            # adv_image = cv2.imread(image_name)
            # adv_image = np.load(image_name)
            # image_basename = os.path.basename(image_name)
            # new_image_basename = image_basename[:-4]+All_Attack_Methods[i]+'.jpg'
            if advImage_number < pickNumber_adv + 1:
                train_path.append(image_name)
                train_ground_label.append(1)
                # cv2.imwrite(os.path.join(train_adv_path,new_image_basename),adv_image)
                # np.save(os.path.join(train_adv_path,new_image_basename),adv_image)
                number1 += 1
            if advImage_number > pickNumber_adv:
                test_path.append(image_name)
                test_ground_label.append(1)
                # cv2.imwrite(os.path.join(test_adv_path,new_image_basename),adv_image)
                # np.save(os.path.join(test_adv_path,new_image_basename),adv_image)
                number2 += 1
            if advImage_number > adv_sample_number - 1:
                logger.info(f"读取{All_Attack_Methods[i]}的对抗样本结束！！！")
                # detect_path.append(image_name)
                # detect_ground_label.append(1)
                break
        adv_total_length = len(adv_file)
        print(f"adv_total_length_{All_Attack_Methods[i]}={adv_total_length}")
        print(f"number1_{All_Attack_Methods[i]}={number1}")
        print(f"number2_{All_Attack_Methods[i]}={number2}")
        time.sleep(10)

    ###读正常样本
    ori_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/all_attack/ori'
    ### 自制正常数据集
    if os.path.isdir(ori_path):
        ori_file = get_image_list(ori_path)
    else:
        ori_file = [ori_path]

    ori_sample_number = len(ori_file)
    logger.info(f"ori_sample_number={ori_sample_number}")
    pickNumber_v1 = adv_sample_number
    pickNumber_ori = pickNumber_adv
    logger.info(f"pickNumber_ori={pickNumber_ori}")
    number3 = 0
    number4 = 0
    random.shuffle(ori_file)
    # 保证ori_file中保存的图片和adv_file中的成对存在
    if All_Attack_Methods[i] == 'backdoor':
        for image_name in ori_file:
            image_files.append(image_name)
            oriImage_number += 1
            if oriImage_number < pickNumber_ori + 1:
                train_path.append(image_name)
                train_ground_label.append(0)
                number3 += 1
            if oriImage_number > pickNumber_ori:
                test_path.append(image_name)
                test_ground_label.append(0)
                number4 += 1
            if oriImage_number > pickNumber_v1 - 1:
                print(f"原始数据读取完毕！！！")
                # detect_path.append(image_name)
                # detect_ground_label.append(0)
                break
    else:
        for image_name in ori_file:
            image_files.append(image_name)
            oriImage_number += 1
            if oriImage_number < pickNumber_ori + 1:
                train_path.append(image_name)
                train_ground_label.append(0)
                number3 += 1
            if oriImage_number > pickNumber_ori:
                test_path.append(image_name)
                test_ground_label.append(0)
                number4 += 1
            if oriImage_number > pickNumber_v1 - 1:
                print(f"原始数据读取完毕！！！")
                # detect_path.append(image_name)
                # detect_ground_label.append(0)
                break
    print(f"number3={number3}")
    print(f"number4={number4}")
    print(f"train_ground_label={len(train_ground_label)},test_ground_label={len(test_ground_label)}")
    print(f"train_path={len(train_path)},test_path={len(test_path)}")
    print(f"读取样本后的总图片数量={len(image_files)}")
    time.sleep(10)

    ### 生成自制训练数据集
    train_sample_number = 0
    train_threshold = 600
    num = int(len(train_path) / 2)
    for i in range(num):

        imageOri_pil = Image.open(train_path[num + i])
        imageAdv_pil = Image.open(train_path[i])

        logger.info(f"num={i},imageOri_pil={imageOri_pil.size}")
        logger.info(f"num={i},imageAdv_pil={imageAdv_pil.size}")

        _, outputs_post_ori = yolos.detect_image(imageOri_pil, save, os.path.basename(train_path[num + i]))
        _, outputs_post_adv = yolos.detect_image(imageAdv_pil, save, os.path.basename(train_path[i]))

        logger.info(f"outputs_post_ori={outputs_post_ori}")
        logger.info(f"outputs_post_adv={outputs_post_adv}")

        closest_box_idx = None
        box_idx_ori = None
        box_idx_adv = None
        # closest_box_idx = []
        closest_distance = float('inf')
        max_distance = 0
        max_distance111 = 0
        max_box_idx = []
        # if outputs_post_adv == '0' or outputs_post_ori == '0':
        if not outputs_post_adv.any() or not outputs_post_ori.any():
            logger.info(f"没检测到物体！！！")
            continue
        train_sample_number += 1
        if train_sample_number > train_threshold:
            logger.info(f"已生成{train_threshold}张训练样本！")
            break
        if All_Attack_Methods[0] == 'backdoor':
            for idx, box_adv in enumerate(outputs_post_adv):
                box_size = max_boxDistance(box_adv)
                if box_size > max_distance:
                    max_distance = box_size
                    box_idx_adv = idx
            for idx, box_ori in enumerate(outputs_post_ori):
                box_size111 = max_boxDistance(box_ori)
                # logger.info(f"box_size111={box_size111}")
                if box_size111 > max_distance111:
                    max_distance111 = box_size
                    box_idx_ori = idx
        else:
            # 找到outputs_post_adv中每一个框与outputs_post_ori中的最接近的框的idx
            # max_box_idx中存储的为ori中框的索引，max_box_idx的下标就是adv中框的索引
            for box_adv in outputs_post_adv:  # 每个框都保存，然后选框最大的一个
                for idx, box_ori in enumerate(outputs_post_ori):
                    distance = box_distance(box_ori[0:4], box_adv[0:4])
                    # logger.info(f"box_ori[0:4]={box_ori[0:4]}")
                    # logger.info(f"box_adv[0:4]={box_adv[0:4]}")
                    # logger.info(f"distance={distance}")
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_box_idx = idx
                max_box_idx.append(closest_box_idx)
                # 选择最大的一个框
            for j in range(len(max_box_idx)):
                max_box_distance = max_boxDistance(outputs_post_ori[max_box_idx[j]][0:4])
                if max_box_distance > max_distance:
                    max_distance = max_box_distance
                    box_idx_ori = max_box_idx[j]
                    box_idx_adv = j

        box1 = outputs_post_ori[box_idx_ori][0:4]
        box2 = outputs_post_adv[box_idx_adv][0:4]
        logger.info(f"box1={box1}")
        logger.info(f"box2={box2}")

        # 图片裁剪和保存
        crop_and_save_image(train_path[num + i], box1,
                            os.path.join(test_image_dir + "/{}/train/ori".format(All_Attack_Methods[0]),
                                         os.path.basename(train_path[num + i])))
        crop_and_save_image(train_path[i], box2,
                            os.path.join(test_image_dir + "/{}/train/adv".format(All_Attack_Methods[0]),
                                         os.path.basename(train_path[i])))
        # exit()
    logger.info(f"训练样本数：{train_sample_number}")
    time.sleep(10)

    ### 生成自制测试数据集
    test_sample_number = 0
    test_threshold = 200
    num111 = int(len(test_path) / 2)
    for i in range(num111):

        imageOri_pil_test = Image.open(test_path[num111 + i])
        imageAdv_pil_test = Image.open(test_path[i])

        _, outputs_post_ori = yolos.detect_image(imageOri_pil_test, save, os.path.basename(test_path[num111 + i]))
        _, outputs_post_adv = yolos.detect_image(imageAdv_pil_test, save, os.path.basename(test_path[i]))

        # if outputs_post_adv == '0' or outputs_post_ori == '0':
        if not outputs_post_adv.any() or not outputs_post_ori.any():
            logger.info(f"没检测到物体！！！")
            continue
        test_sample_number += 1
        if test_sample_number > test_threshold:
            logger.info(f"以生成{test_threshold}张测试样本！")
            break
        closest_box_idx = None
        box_idx_ori = None
        box_idx_adv = None
        closest_distance = float('inf')
        max_distance = 0
        max_box_idx = []
        max_distance111 = 0

        if All_Attack_Methods[0] == 'backdoor':
            for idx, box_adv in enumerate(outputs_post_adv):
                box_size = max_boxDistance(box_adv)
                if box_size > max_distance:
                    max_distance = box_size
                    box_idx_adv = idx
            for idx, box_ori in enumerate(outputs_post_ori):
                box_size111 = max_boxDistance(box_ori)
                # logger.info(f"box_size111={box_size111}")
                if box_size111 > max_distance111:
                    max_distance111 = box_size
                    box_idx_ori = idx

        else:
            # 找到outputs_post_adv中每一个框与outputs_post_ori中的最接近的框的idx
            # max_box_idx中存储的为ori中框的索引，max_box_idx的下标就是adv中框的索引
            for box_adv in outputs_post_adv:  # 每个框都保存，然后选框最大的一个
                for idx, box_ori in enumerate(outputs_post_ori):
                    distance = box_distance(box_ori[0:4], box_adv[0:4])
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_box_idx = idx
                max_box_idx.append(closest_box_idx)
                # 选框最大的一个框
            for j in range(len(max_box_idx)):
                max_box_distance = max_boxDistance(outputs_post_ori[max_box_idx[j]][0:4])
                if max_box_distance > max_distance:
                    max_distance = max_box_distance
                    box_idx_ori = max_box_idx[j]
                    box_idx_adv = j

        box1 = outputs_post_ori[box_idx_ori][0:4]
        box2 = outputs_post_adv[box_idx_adv][0:4]
        # 图片裁剪和保存
        crop_and_save_image(test_path[num111 + i], box1,
                            os.path.join(test_image_dir + "/{}/test/ori".format(All_Attack_Methods[0]),
                                         os.path.basename(test_path[num111 + i])))
        crop_and_save_image(test_path[i], box2,
                            os.path.join(test_image_dir + "/{}/test/adv".format(All_Attack_Methods[0]),
                                         os.path.basename(test_path[i])))
    logger.info(f"测试样本数：{test_sample_number}")

    logger.info(f"train_num={len(train_path)},test_num={len(test_path)}")

    logger.info(f"****开始训练*******")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    # Initialize the dataset
    dataset = datasets.ImageFolder(root=test_image_dir + '/{}/train'.format(All_Attack_Methods[0]), transform=transform)
    test_dataset = datasets.ImageFolder(root=test_image_dir + '/{}/test'.format(All_Attack_Methods[0]),
                                        transform=transform)
    logger.info(f"dataset={len(dataset)},test_dataset={len(test_dataset)}")

    # Define batch size for DataLoader
    batch_size = 10

    # Initialize the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 模型训练
    # model=MLP().to(device)
    # model=CNNClassifier().to(device)
    # model=linear_model(714000,2).to(device)
    # model = SimpleCNN().to(device)
    # model = CNN().to(device)
    model = binary_model(2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    logger.info(f"---training------")
    train_epochs = 20

    ###打乱训练集合测试集
    # zipped_train_data = list(zip(train_path,train_ground_label))
    # random.shuffle(zipped_train_data)
    # shuffled_train_path, shuffled_train_ground_label = zip(*zipped_train_data)
    # zipped_test_data = list(zip(test_path,test_ground_label))
    # random.shuffle(zipped_test_data)
    # shuffled_test_path, shuffled_test_ground_label = zip(*zipped_test_data)
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
        if Test_Accuarcy > Best_Acc and Test_Accuarcy > 0.5:
            Best_Acc = Test_Accuarcy
            logger.info(f"成功保存！！！")
            torch.save(model.state_dict(),
                       '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/yolo_detect/model_data/{}_CNN_Classifier.pth'.format(
                           All_Attack_Methods[0]))
            torch.save(model.state_dict(),
                       '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/yolo_detect/model_data/{}_CNN_Classifier_{}.pth'.format(
                           All_Attack_Methods[0], Test_Accuarcy))
    logger.info(f"All_Attack_Methods={All_Attack_Methods[0]}")


def CNN_Classifier_Detect(test_image_dir, Attack_Method, Method, ori_path, yolos):
    save = True
    # 建立目录
    attack_methods = ['FGSM',  'MIFGSM',  'patch_attack', 'GuassianBlurAttack']
    for i in range(len(attack_methods)):
        os.makedirs(test_image_dir + "/DetectData/{}/ori".format(attack_methods[i]), exist_ok=True)
        os.makedirs(test_image_dir + "/DetectData/{}/adv".format(attack_methods[i]), exist_ok=True)

    image_files = []
    # 检测集
    detect_path = []
    detect_ground_label = []

    pickNumber = 3000

    Attack_Method='GuassianBlurAttack'
    random.seed(7)
    ###读取对抗样本
    # All_Attack_Methods=['FGSM','IGSM','MIFGSM','PGD','patch_attack','GuassianBlurAttack','backdoor']

    # All_Attack_Methods = ['GuassianBlurAttack']
    # All_Attack_Methods = ['backdoor']
    advImage_number = 0
    # adv_path='/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true/{}'.format(All_Attack_Methods[i])
    if Attack_Method == 'backdoor':
        adv_path = '/data1/BigPlatform/ZJPlatform/bice_pingce/002-Code/001_ObjectDetection/002_Code/extra_task/yolox_new/backdoor/data/jpg/'
    else:
        adv_path = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/{}".format(
            Attack_Method)

    if os.path.isdir(adv_path):
        adv_file = get_image_list(adv_path)
    else:
        adv_file = [adv_path]
    random.shuffle(adv_file)
    for image_name in adv_file:
        image_files.append(image_name)
        advImage_number += 1
        # adv_image = cv2.imread(image_name)
        # adv_image = np.load(image_name)
        # image_basename = os.path.basename(image_name)
        # new_image_basename = image_basename[:-4]+All_Attack_Methods[i]+'.jpg'
        if advImage_number < pickNumber + 1:
            detect_path.append(image_name)
            detect_ground_label.append(1)
        if advImage_number > pickNumber - 1:
            print(f"读取{Attack_Method}的对抗样本结束！！！")
            # detect_path.append(image_name)
            # detect_ground_label.append(1)
            break
    adv_total_length = len(adv_file)
    print(f"adv_total_length_{Attack_Method}={adv_total_length}")

    ###读正常样本
    ori_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/all_attack/ori'
    if os.path.isdir(ori_path):
        ori_file = get_image_list(ori_path)
    else:
        ori_file = [ori_path]

    ori_total_length = len(ori_file)
    print(f"ori_total_length={ori_total_length}")
    if pickNumber > ori_total_length:
        pickNumber = ori_total_length
        print(f"pickNumber过大，超过了ori_total_length={ori_total_length}")

    oriImage_number = 0

    if Attack_Method == 'backdoor':
        for image_name in ori_file:
            image_files.append(image_name)
            oriImage_number += 1
            if oriImage_number < pickNumber + 1:
                detect_path.append(image_name)
                detect_ground_label.append(0)
            if oriImage_number > pickNumber - 1:
                print(f"原始数据读取完毕！！！")
                # detect_path.append(image_name)
                # detect_ground_label.append(0)
                break
    else:
        for i in range(len(adv_file)):
            base_name = os.path.basename(adv_file[i])
            image_name = os.path.join(ori_path, base_name)
            if image_name in ori_file:
                image_files.append(image_name)
                oriImage_number += 1
                if oriImage_number < pickNumber + 1:
                    detect_path.append(image_name)
                    detect_ground_label.append(0)
                if oriImage_number > pickNumber - 1:
                    print(f"原始数据读取完毕！！！")
                    # detect_path.append(image_name)
                    # detect_ground_label.append(0)
                    break

    print(f"detect_ground_label={len(detect_ground_label)},detect_path={len(detect_path)}")
    print(f"读取样本后的总图片数量={len(image_files)}")
    time.sleep(10)

    val_number = len(get_image_list(os.path.join(test_image_dir + "/DetectData/{}/ori".format(Attack_Method))))
    if val_number == 0:
        ### 生成自制验证数据集
        val_sample_number = 0
        val_threshold = 200
        num222 = int(len(detect_path) / 2)
        for i in range(num222):

            imageOri_pil_detect = Image.open(detect_path[num222 + i])
            imageAdv_pil_detect = Image.open(detect_path[i])

            _, outputs_post_ori = yolos.detect_image(imageOri_pil_detect, save,
                                                     os.path.basename(detect_path[num222 + i]))
            _, outputs_post_adv = yolos.detect_image(imageAdv_pil_detect, save, os.path.basename(detect_path[i]))
            logger.info(f"detect_path[num222+i]={detect_path[num222 + i]}")
            logger.info(f"detect_path[i]={detect_path[i]}")
            # if outputs_post_adv == '0' or outputs_post_ori == '0':
            if not outputs_post_adv.any() or not outputs_post_ori.any():
                logger.info(f"没检测到物体！！！")
                continue
            val_sample_number += 1
            if val_sample_number > val_threshold:
                logger.info(f"已保存{val_threshold}张验证样本！")
                break
            closest_box_idx = None
            box_idx_ori = None
            box_idx_adv = None
            # closest_box_idx = []
            closest_distance = float('inf')
            max_distance = 0
            max_distance111 = 0
            max_box_idx = []
            for idx, box_adv in enumerate(outputs_post_adv):
                box_size = max_boxDistance(box_adv)
                if box_size > max_distance:
                    max_distance = box_size
                    box_idx_adv = idx
            for idx, box_ori in enumerate(outputs_post_ori):
                box_size111 = max_boxDistance(box_ori)
                logger.info(f"box_size111={box_size111}")
                if box_size111 > max_distance111:
                    max_distance111 = box_size
                    box_idx_ori = idx

            # 找到outputs_post_adv中每一个框与outputs_post_ori中的最接近的框的idx
            # max_box_idx中存储的为ori中框的索引，max_box_idx的下标就是adv中框的索引
            # for box_adv in outputs_post_adv: # 每个框都保存，然后选框最大的一个
            # for idx,box_ori in enumerate(outputs_post_ori):
            # distance = box_distance(box_ori[0:4], box_adv[0:4])
            # if distance < closest_distance:
            # closest_distance = distance
            # closest_box_idx = idx
            # max_box_idx.append(closest_box_idx)
            # # 选框最大的一个框
            # for j in range(len(max_box_idx)):
            # max_box_distance = max_boxDistance(outputs_post_ori[max_box_idx[j]][0:4])
            # if max_box_distance > max_distance:
            # max_distance = max_box_distance
            # box_idx_ori = max_box_idx[j]
            # box_idx_adv = j
            logger.info(f"outputs_post_ori={outputs_post_ori}")
            logger.info(f"outputs_post_adv={outputs_post_adv}")
            logger.info(f"box_idx_ori={box_idx_ori}")
            box1 = outputs_post_ori[box_idx_ori]
            box2 = outputs_post_adv[box_idx_adv]
            logger.info(f"box1={box1}")
            # 图片裁剪和保存
            crop_and_save_image(detect_path[num222 + i], box1,
                                os.path.join(test_image_dir + "/DetectData/{}/ori".format(Attack_Method),
                                             os.path.basename(detect_path[num222 + i])))
            crop_and_save_image(detect_path[i], box2,
                                os.path.join(test_image_dir + "/DetectData/{}/adv".format(Attack_Method),
                                             os.path.basename(detect_path[i])))

    val_number = len(get_image_list(os.path.join(test_image_dir + "/DetectData/{}/ori".format(Attack_Method))))
    logger.info(f"验证集样本数：{val_number}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # Initialize the dataset
    test_dataset = datasets.ImageFolder(root=test_image_dir + '/DetectData/{}'.format(Attack_Method),
                                        transform=transform)

    # Define batch size for DataLoader
    batch_size = 10

    # Initialize the DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # model = CNN().to(device)
    # model = SimpleCNN().to(device)
    # model = MLP().to(device)
    # model = CNNClassifier().to(device)
    model = binary_model(2).to(device)

    model_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/yolo_detect/model_data/ALL_warship_CNN_Classifier.pth'
    model.load_state_dict(torch.load(model_path))

    test_correct = 0
    total111 = 0
    model.eval()
    print(f"test_dataset={len(test_dataset)}")

    # test_loader=random.sample(list(test_loader),200)
    start_time = time.time()
    for images111, labels111 in test_loader:
        images111, labels111 = images111.to(device), labels111.to(device)
        outputs111 = model(images111)
        predicted_test = torch.argmax(outputs111.data, 1)
        test_correct += (predicted_test == labels111).sum().item()
        total111 += images111.shape[0]
        # if total111 > 199:
        # break
    end_time = time.time()
    everage_time = (end_time - start_time) / len(test_dataset)
    Test_Accuarcy = test_correct / total111
    logger.info(f"CNN_Classifier_Detect:Test_Accuarcy={test_correct}/{total111}={test_correct / total111}")
    logger.info(f"everage_time={everage_time},Attack_Method={Attack_Method}")

    abnormal_file_dir = os.path.join(test_image_dir, 'DetectData')
    abnormal_file_path = os.path.join(abnormal_file_dir, Attack_Method)

    jsontext = {
        "Abnormal_File_Path": abnormal_file_path,
        'Abnormal_File_Name': Attack_Method,
        "Method": Method,
        'Score': round(Test_Accuarcy, 2),
        "Abnormal_File_Dir": abnormal_file_dir,
    }
    # print(jsontext)
    return jsontext


def detect_main(detparams):
    print(f"device={device}")

    yolos = YOLO()
    # ssd_Model = ssd.to(device).eval()
    # print(f"ssd_Model={ssd_Model}")
    Attack_Method = detparams['Attack_Method']
    Method = detparams['Method']
    ori_path = detparams['DatasetDIR']
    test_image_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/GuassianBlurAttack'

    saved_model_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/yolo_detect/model_data/{}_CNN_Classifier.pth'.format(
        Attack_Method)
    if os.path.exists(saved_model_path):
        logger.info(f"开始检测！！！")
        result = CNN_Classifier_Detect(test_image_dir, Attack_Method, Method, ori_path, yolos)
    else:
        logger.info(f"开始训练二分类器！！！")
        train_binary_classifier(test_image_dir, Attack_Method, ori_path, yolos)
        logger.info(f"开始检测！！！")
        result = CNN_Classifier_Detect(test_image_dir, Attack_Method, Method, ori_path, yolos)

    return result


if __name__ == "__main__":
    detparams = {
        # 前后端传输
        'TaskId': '12378',
        'DatasetDIR': '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/all_attack_v2',
        'ModelDIR': 'yolox_ship.pth',
        'Method': 'PrePro_CNNClassifier',
        'Attack_Method': 'ALL_warship_v2'

    }

    initial_gpu_max_usage = torch.cuda.max_memory_allocated()
    before_utilization = get_gpu_utilization()
    start_time = time.time()
    result = detect_main(detparams)
    end_time = time.time()
    usetime = end_time - start_time
    after_utilization = get_gpu_utilization()
    final_gpu_max_usage = torch.cuda.max_memory_allocated()
    max_gpu_usage_change = final_gpu_max_usage - initial_gpu_max_usage
    gpu_load = torch.cuda.get_device_properties(device.index).total_memory
    print(f'gpu_load: {gpu_load}')
    print(f"Before main: GPU Utilization = {before_utilization}%")
    print(f"After main: GPU Utilization = {after_utilization}%")
    print(f'used time: {usetime}s')
    print(f'GPU占用率: {sum([abs(a - b) for a, b in zip(after_utilization, before_utilization)])}%')
    print(f"最大显存占用率: {max_gpu_usage_change / gpu_load:.4f} %")

    print(result)
