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
import shutil
import torchvision.models as models
from sdd_zz import SSD
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)

os.environ["CUDA_VISABLE_DEVICES"] = "1,2,3,4,5,6,7"
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssd = SSD()
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


def tensor_to_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  ###去掉batch维度
    tensor = tensor.permute(1, 2, 0)  ##将c,h,w 转换为h,w,c
    tensor = tensor.mul(255).clamp(0, 255)  ###将像素值转换为0-255之间
    tensor = tensor.cpu().numpy().astype('uint8')  ###
    return tensor

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
    #
    # new_list = [int(value * 255) for value in conf]
    #
    # width, height = cro_image.size
    # pixel_values_triplicated = conf * 3
    # for i in range(3):
    #     for j, pixel_value in enumerate(pixel_values_triplicated[i * len(conf):(i + 1) * len(conf)]):
    #         cro_image.putpixel((width - 1 - len(conf) + j, height - 1),
    #                            (int(pixel_value * 255), int(pixel_value * 255), int(pixel_value * 255)))

    return cro_image


def ssd_detection(img_file, model):
    imageOri_pil = Image.open(img_file)
    _, outputs_post_ori, outputs_conf_ori = ssd.detect_image(imageOri_pil)

    # 定义变量
    box_idx_ori = None
    max_distance = 0

    # 检测结果是否存在物体
    if outputs_post_ori == '0':
        logger.info(f"没检测到物体！！！")
        return 0

    for idx, box_ori in enumerate(outputs_post_ori):
        box_size = max_boxDistance(box_ori)
        if box_size > max_distance:
            max_distance = box_size
            box_idx_ori = idx

    # 获取框信息
    box1 = outputs_post_ori[box_idx_ori][0:4]
    conf1 = outputs_conf_ori[box_idx_ori]

    # 图片裁剪和保存
    img = crop_and_save_image(imageOri_pil, box1, conf1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((80, 80)),
    ])

    transformed_image = transform(img).unsqueeze(0)
    output = model(transformed_image)
    predicted_test = torch.argmax(output.data, 1)
    return predicted_test.numpy()[0]


if __name__ == "__main__":
    FGSM_IMAGES_PATH = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/VOC_adv_image_true/patch_attack"  # 测试数据集路径

    DETECTOR_PATH = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/ssd_detect/model_data/ALL_VOC_CNN_Classifier_1.0.pth'

    detector = model_load(DETECTOR_PATH)

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
                    result = ssd_detection(img_file, detector)
                    end_time = time.time()
                    results.append(result)

                    # print(file_path)
                    # if 'ori' in file_path.split('/') and result ==1:
                    #     shutil.copy(file_path,  os.path.join('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/VOC_adv_image_true/select/patch_attack/ori', os.path.basename(file)))
                    # elif'ori' in file_path.split('/') and result == 0:
                    #     print("良性判断错了")
                    # if 'adv' in file_path.split('/') and result == 0:
                    #     shutil.copy(file_path,  os.path.join('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/VOC_adv_image_true/select/patch_attack/adv', os.path.basename(file)))
                    # elif 'adv' in file_path.split('/') and result == 1:
                    #     print("恶意判断错了")

                    total_time += (end_time - start_time)

    correct_predictions = sum(p == l for p, l in zip(results, labels))
    accuracy = correct_predictions / len(labels)
    avg_time = total_time / len(labels)
    print("ACC: " + str(accuracy))
    print("avg_time: " + str(avg_time))
    # print(results)
    # print(labels)
