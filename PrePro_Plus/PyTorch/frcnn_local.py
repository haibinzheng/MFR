import torch
import numpy as np
import torch.optim as optim
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms

import foolbox
import matplotlib.pyplot as plt

from art.defences.preprocessor import SpatialSmoothing
import foolbox as fb
from foolbox.models import PyTorchModel
from foolbox.utils import accuracy
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

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import torch.nn as nn
import argparse
from loguru import logger
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
import time
import torchvision.transforms.functional as TF
import random
import torchvision.models as models
from frcnn_plus_zz_poi import FRCNN

from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         preprocess_input, show_config)

import shutil

frcnn = FRCNN()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)  # 使用ResNet18作为基础模型
        self.fc = nn.Linear(1000, 2)  # 二分类任务，输出维度为2

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


def model_load(path):
    model = CNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# 找到最大的一个框
def max_boxDistance(box):
    x1, y1, x2, y2 = box
    distance = int(math.fabs((x2 - x1) * (y2 - y1)))
    return distance


# 图片裁剪和保存
def crop_and_save_image(image, box, conf):
    conf = conf[:len(conf) - 1]

    box[0] = max(box[0], 0)
    box[1] = max(box[1], 0)
    box[2] = min(box[2], image.size[1])
    box[3] = min(box[3], image.size[0])
    x1, y1, x2, y2 = box.tolist()  # top,left,bottom,right
    a1, a2, b1, b2 = math.ceil(x1), math.ceil(y1), int(math.fabs(x2 - x1)), int(math.fabs(y2 - y1))

    cropped_image = image.crop([y1, x1, y2, x2])
    new_size = (80, 80)
    cro_image = cropped_image.resize(new_size, Image.ANTIALIAS)
    # 转化为0-255
    # new_list = [int(value * 255) for value in conf]
    #
    # width, height = cro_image.size
    # pixel_values_triplicated = conf * 3
    # for i in range(3):
    #     for j, pixel_value in enumerate(pixel_values_triplicated[i * len(conf):(i + 1) * len(conf)]):
    #         cro_image.putpixel((width - 1 - len(conf) + j, height - 1), (int(pixel_value * 255)))
    return cro_image


def frcnn_detection(img_file, model):
    imageOri_pil = Image.open(img_file)
    _, outputs_post_ori, output_conf_ori = frcnn.detect_image(imageOri_pil)

    box_idx_ori = None
    max_distance_ori = 0

    # 检测是否有框存在
    if outputs_post_ori == '0':
        return 0

    # **原图**中选出最大的框
    for idx, box_ori in enumerate(outputs_post_ori):
        box_size = max_boxDistance(box_ori)
        if box_size > max_distance_ori:
            max_distance_ori = box_size
            box_idx_ori = idx

    # 提取框和置信度
    box1 = outputs_post_ori[box_idx_ori][0:4]
    conf1 = output_conf_ori[box_idx_ori]

    # 图片裁剪和保存
    img = crop_and_save_image(imageOri_pil, box1, conf1)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 将灰度图扩展为 3 通道
        transforms.ToTensor(),
        transforms.Resize((80, 80)),
    ])

    transformed_image = transform(img).unsqueeze(0)

    output = model(transformed_image)
    predicted_test = torch.argmax(output.data, 1)
    return predicted_test.numpy()[0]


if __name__ == "__main__":
    FGSM_IMAGES_PATH = "/data/Newdisk/chenjingwen/DT_B4/IndicatorsCoverage/SJS/PrePro_Plus/FRCNN_FLIR/dataset/FLIR/backdoor_less"  # 测试数据集路径

    DETECTOR_PATH = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/frcnn_detect/model_data/backdoor_CNN_classifier.pth'

    detector = model_load(DETECTOR_PATH)
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    results = []
    labels = []
    total_time = 0
    adv_saved = 0
    ori_saved = 0
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
                    result = frcnn_detection(img_file, detector)
                    end_time = time.time()
                    results.append(result)
                    print(file_path)
                    # if 'ori' in file_path.split('/') and result == 1 and ori_saved < 50:
                    #     ori_saved += 1
                    #     shutil.copy(file_path, os.path.join(
                    #         '/data/Newdisk/chenjingwen/DT_B4/IndicatorsCoverage/SJS/PrePro_Plus/FRCNN_FLIR/dataset/FLIR/backdoor_less/ori',
                    #         os.path.basename(file)))
                    # elif'ori' in file_path.split('/') and result == 0:
                    #     if random.random() < 0.05:
                    #         shutil.copy(file_path, os.path.join(
                    #             '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/FILR_adv_true_mwb/select/FGSM/ori',
                    #             os.path.basename(file)))
                    #     print("良性判断错了")
                    # if 'poi' in file_path.split('/') and result == 0 and adv_saved < 50:
                    #     adv_saved += 1
                    #     shutil.copy(file_path, os.path.join(
                    #         '/data/Newdisk/chenjingwen/DT_B4/IndicatorsCoverage/SJS/PrePro_Plus/FRCNN_FLIR/dataset/FLIR/backdoor_less/poi',
                    #         os.path.basename(file)))
                    # elif 'adv' in file_path.split('/') and result == 1:
                    #     if random.random() < 0.05:
                    #         shutil.copy(file_path, os.path.join(
                    #             '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/FILR_adv_true_mwb/select/FGSM/adv',
                    #             os.path.basename(file)))
                    #     print("恶意判断错了")

                    total_time += (end_time - start_time)

    correct_predictions = sum(p == l for p, l in zip(results, labels))
    accuracy = correct_predictions / len(labels)
    avg_time = total_time / len(labels)
    print("ACC: " + str(accuracy))
    print("avg_time: " + str(avg_time))
    # print(results)
    # print(labels)
