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

import torchvision.models as models
from ceshi import FRCNN

from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         preprocess_input, show_config)

import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)  # 使用ResNet18作为基础模型
        self.fc = nn.Linear(1000, 2)  # 二分类任务，输出维度为2

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


img_test_size = (640, 640)


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


def crop_and_save_image(image_path, box, conf):
    # print(box)
    # print(conf)
    image = Image.open(image_path)

    box[0] = max(box[0], 0)
    box[1] = max(box[1], 0)
    box[2] = min(box[2], image.size[1])
    box[3] = min(box[3], image.size[0])

    x1, y1, x2, y2 = box.tolist()  # top,left,bottom,right
    a1, a2, b1, b2 = math.ceil(x1), math.ceil(y1), int(math.fabs(x2 - x1)), int(math.fabs(y2 - y1))

    cropped_image = image.crop([y1, x1, y2, x2])
    new_size = (80, 80)
    cro_image = cropped_image.resize(new_size, Image.ANTIALIAS)

    # print(f"ori_cro_image={np.array(cro_image)}")
    # print(f"ori_cro_image={np.array(cro_image).shape}")

    # width, height = cro_image.size
    # pixel_values_triplicated = conf * 3
    # for x in range(width):
    #     for c in range(3):  # 对每个通道进行操作
    #         # 获取最后一行的像素值并乘以置信度
    #         pixel_value = cro_image.getpixel((x, height - 1))[c]
    #         new_pixel_value = int(pixel_value * conf)
    #         # 更新像素值
    #         cro_image.putpixel((x, height - 1), tuple(
    #             int(cro_image.getpixel((x, height - 1))[i] * conf) for i in range(3)
    #         ))
    return cro_image


def frcnn_detection(img_file, detector, frcnn):
    img = Image.open(img_file)
    _, top_boxes, top_confs = frcnn.detect_image(img, False, None)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((80, 80)),
    ])
   
    if isinstance(top_boxes[0], int):
        # logger.info(f"outputs_post_adv is None")
        # return 0
        transformed_image = transform(img).unsqueeze(0)
        output = detector(transformed_image)
        predicted_test = torch.argmax(output.data, 1)
        return predicted_test.numpy()[0]

    box_idx_adv = None
    max_distance = 0
    max_box_idx = None

    # 遍历对抗样本中的所有框，找到最大框距离的框
    for idx, box_adv in enumerate(top_boxes):

        max_box_distance = max_boxDistance(box_adv[0:4])  # 计算当前框的“框距离”
        if max_box_distance > max_distance:
            max_distance = max_box_distance
            box_idx_adv = idx

    # 提取框信息

    box1 = top_boxes[box_idx_adv][0:4]  # 最大框距离的框的坐标
    conf1 = top_confs[box_idx_adv]  # 对应的置信度

    img = crop_and_save_image(img_file, box1, conf1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((80, 80)),
    ])

    transformed_image = transform(img).unsqueeze(0)
    output = detector(transformed_image)
    predicted_test = torch.argmax(output.data, 1)
    print(predicted_test.numpy()[0])

    return predicted_test.numpy()[0]


def detect_main():
    frcnn = FRCNN()
    print(f"device={device}")

    labels = []
    results = []
    total_time = 0
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    test_image_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_frcnn/select/patch_attack'

    detector_path = '//data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/frcnn_detect/model_data/ALL_v2_warship_CNN_Classifier.pth'
    # detector = SimpleCNN()
    detector = CNN()
    detector.load_state_dict(torch.load(detector_path))
    detector.eval()
    if os.path.exists(test_image_dir):
        logger.info(f"开始检测！！！")
        for root, dirs, files in os.walk(test_image_dir):  # 以该数据集为例，可替换

            for file in files:
                if file.lower().endswith(supported_extensions):
                    file_path = os.path.join(root, file)
                    print(file_path)

                    if 'ori' in file_path.split('/'):
                        labels.append(1)
                    else:
                        labels.append(0)

                    start_time = time.time()

                    result = frcnn_detection(file_path, detector, frcnn)
                    end_time = time.time()
                    results.append(result)

                    total_time += (end_time - start_time)

                    # if 'ori' in file_path.split('/') and result == 1:
                    #     shutil.copy(file_path, os.path.join(
                    #         '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_frcnn/select/patch_attack/ori',
                    #         os.path.basename(file)))
                    # elif 'ori' in file_path.split('/') and result == 0:
                    #     # if random.random() < 0.05:
                    #     #     shutil.copy(file_path, os.path.join(
                    #     #         '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/select/patch_attack/ori',
                    #     #         os.path.basename(file)))
                    #     print("良性判断错了")
                    # if 'adv' in file_path.split('/') and result == 0:
                    #     shutil.copy(file_path, os.path.join(
                    #         '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_frcnn/select/patch_attack/adv',
                    #         os.path.basename(file)))
                    # elif 'adv' in file_path.split('/') and result == 1:
                    #     # if random.random() < 0.05:
                    #     #     shutil.copy(file_path, os.path.join(
                    #     #         '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/select/patch_attack/adv',
                    #     #         os.path.basename(file)))
                    #     print("恶意判断错了")

    # batch_size = 32
    # dataloader = prepare_data(results, labels, batch_size=batch_size)
    #
    # # 初始化模型
    # model = SimpleCNN()
    #
    # # 训练模型
    # trained_model = train_model(model, dataloader, num_epochs=10)
    correct_predictions = sum(p == l for p, l in zip(results, labels))
    accuracy = correct_predictions / len(labels)
    avg_time = total_time / len(labels)
    print("ACC: " + str(accuracy))
    print("avg_time: " + str(avg_time))


if __name__ == "__main__":
    result = detect_main()
    print(result)
