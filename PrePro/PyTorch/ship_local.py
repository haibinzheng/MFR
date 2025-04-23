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
import shutil
import subprocess


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
    # image1 = Image.open(image_path)
    # img = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR) #PIL转为opencv
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
    # logger.info(f"-----裁剪完成------")


def yolox_detection(img_file, Method, detector, yolos):
    img = Image.open(img_file)
    _, top_boxes, top_confs = yolos.detect_image(img, False, None)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    if len(top_boxes) == 0:
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
        transforms.Resize((224, 224)),
    ])

    transformed_image = transform(img).unsqueeze(0)
    output = detector(transformed_image)
    predicted_test = torch.argmax(output.data, 1)

    return predicted_test.numpy()[0]


def detect_main(detparams):
    print(f"device={device}")

    yolos = YOLO()

    labels = []
    results = []
    total_time = 0
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    Attack_Method = detparams['Attack_Method']
    Method = detparams['Method']

    test_image_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/select/UAP'

    detector_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/yolo_detect/model_data/ALL_warship_v2_CNN_Classifier.pth'
    # detector = SimpleCNN()
    detector = binary_model(2)
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

                    result = yolox_detection(file_path, Method, detector, yolos)
                    end_time = time.time()
                    results.append(result)

                    total_time += (end_time - start_time)
                    # if 'ori' in file_path.split('/') and result == 1:
                    #     shutil.copy(file_path, os.path.join(
                    #         '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/select/MIFGSM/ori',
                    #         os.path.basename(file)))
                    # elif 'ori' in file_path.split('/') and result == 0:
                    #     if random.random() < 0.05:
                    #         shutil.copy(file_path, os.path.join(
                    #             '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/select/MIFGSM/ori',
                    #             os.path.basename(file)))
                    #     print("良性判断错了")
                    # if 'adv' in file_path.split('/') and result == 0:
                    #     shutil.copy(file_path, os.path.join(
                    #         '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/select/MIFGSM/adv',
                    #         os.path.basename(file)))
                    # elif 'adv' in file_path.split('/') and result == 1:
                    #     if random.random() < 0.05:
                    #         shutil.copy(file_path, os.path.join(
                    #             '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_warship_adv/select/MIFGSM/adv',
                    #             os.path.basename(file)))
                    #     print("恶意判断错了")

    
    correct_predictions = sum(p == l for p, l in zip(results, labels))
    accuracy = correct_predictions / len(labels)
    avg_time = total_time / len(labels)
    print("ACC: " + str(accuracy))
    print("avg_time: " + str(avg_time))


if __name__ == "__main__":
    detparams = {
        # 前后端传输
        'TaskId': '12378',
        'ModelDIR': 'yolox_ship.pth',
        'Method': 'PrePro_CNNClassifier',
        'Attack_Method': 'ALL_warship_v2'

    }
    result = detect_main(detparams)
    print(result)
