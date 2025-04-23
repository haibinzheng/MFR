import sys

import tensorflow as tf
from nets.alexnet import AlexNet_v1, AlexNet_v2
import numpy as np
import os
from loguru import logger
import random
from sklearn.neighbors import KernelDensity
import glob
from joblib import dump, load
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from tensorflow.keras.models import Model
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import shutil

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy", ".PNG", '.JPEG']
im_width, im_height = 224, 224


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 3 * 3, 32)  # 输入 64 * 3 * 3 是因为经过卷积和池化后的尺寸
        self.fc2 = nn.Linear(32, 2)  # 二分类，所以输出是2个节点

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.reshape(1, 128, 6, 6))))  # 卷积 -> ReLU -> 池化
        x = x.view(-1, 64 * 3 * 3)  # 展平
        x = F.relu(self.fc1(x))  # 全连接层1
        x = self.fc2(x)  # 全连接层2
        return x


def model_load(model_path, type):
    if (type == 'classify'):
        model = AlexNet_v1(num_classes=10)
        model.load_weights(model_path)

    if (type == 'detect'):
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()

    return model


def vgg_fgsm_detection(image, model):
    # print("local......................")
    # print(image)
    img = Image.open(image)
    detection_model = model[0]
    classification_model = model[1]

    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((im_width, im_height))
    img = np.array(img) / 255.
    img = np.expand_dims(img, 0)

    output = classification_model(img, training=False)
    pred = tf.argmax(output, axis=1)

    # input = torch.tensor(feature[-2].numpy()).unsqueeze(0)
    # output = detection_model(input)  # 前向传播
    # 
    # predicted_label = torch.argmax(output).item()
    return pred.numpy()[0]


if __name__ == "__main__":
    FGSM_IMAGES_PATH_ori = "/data/Newdisk/chenjingwen/DT_B4/GD_detect/image_tf/dataset/military/AlexNet/FGSM_less_less/ori"  # 测试数据集路径
    FGSM_IMAGES_PATH_adv = "/data/Newdisk/chenjingwen/DT_B4/GD_detect/image_tf/dataset/military/AlexNet/FGSM_less_less/adv"  # 测试数据集路径
    ALEXNET_MODEL_PATH = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image_tf/save_model/military/alexnet_0.6307123899459839.ckpt'
    FGSM_MODEL_PATH = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image_tf/save_model/SentimentDetector_AlexNet_Military_FGSM-Lambda1_1.0.pth"  # 测试检测模型路径

    model = []
    FGSM_MODEL = model_load(FGSM_MODEL_PATH, "detect")
    ALEXNET_MODEL = model_load(ALEXNET_MODEL_PATH, "classify")
    model.append(FGSM_MODEL)
    model.append(ALEXNET_MODEL)
    attack_name = 'FGSM-Lambda1'

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    results = []
    labels = []
    total_time = 0
    total_num = 0
    ori_correct = 0
    adv_correct = 0
    ori_saved = 0
    adv_saved = 0
    for root, dirs, files in os.walk(FGSM_IMAGES_PATH_ori):  # 以该数据集为例，可替换

        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)

                class_name = int(os.path.basename(root))
                labels.append(class_name)
                target_path = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image_tf/dataset/military/AlexNet/FGSM_less_less'

                with open(file_path, "rb") as img_file:
                    start_time = time.time()
                    class_pred = vgg_fgsm_detection(img_file, model)
                    end_time = time.time()
                    results.append(class_pred)
                    ori_saved += 1


                    # if 'ori' in file_path and class_name == class_pred:
                    #     ori_folder = os.path.join(target_path, 'ori')
                    #     if not os.path.exists(ori_folder + '/' + str(class_name)):  # 如果子文件夹不存在，创建它
                    #         os.makedirs(ori_folder + '/' + str(class_name))
                    #     target_path = ori_folder + '/' + str(class_name)  # 设置目标路径为'ori'文件夹
                    #     if os.path.isfile(file_path):
                    #         shutil.copy(file_path, os.path.join(target_path, file))
                    #         ori_saved+=1

                    total_time += (end_time - start_time)

    correct_predictions = sum(p == l for p, l in zip(results, labels))
    ori_accuracy = correct_predictions / len(labels)
    avg_time = total_time / len(labels)
    print("TensorFlow -FGSM -ORI ACC: " + str(ori_accuracy))

    results = []
    labels = []
    for root, dirs, files in os.walk(FGSM_IMAGES_PATH_adv):  # 以该数据集为例，可替换

        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)

                class_name = int(os.path.basename(root))
                labels.append(class_name)
                target_path = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image_tf/dataset/military/AlexNet/FGSM_less_less'

                with open(file_path, "rb") as img_file:
                    start_time = time.time()
                    class_pred = vgg_fgsm_detection(img_file, model)
                    end_time = time.time()
                    results.append(class_pred)
                    adv_saved += 1
                    # if 'adv' in file_path and class_name != class_pred:
                    #     adv_folder = os.path.join(target_path, 'adv')
                    #     if not os.path.exists(adv_folder + '/' + str(class_name)):  # 如果子文件夹不存在，创建它
                    #         os.makedirs(adv_folder + '/' + str(class_name))
                    #     target_path = adv_folder + '/' + str(class_name)  # 设置目标路径为'ori'文件夹
                    #     if os.path.isfile(file_path):
                    #         shutil.copy(file_path, os.path.join(target_path, file))
                    #         adv_saved += 1
                    total_time += (end_time - start_time)

    correct_predictions = sum(p == l for p, l in zip(results, labels))
    adv_accuracy = correct_predictions / len(labels)
    avg_time = total_time / len(labels)
    print("TensorFlow -FGSM -adv ACC: " + str(adv_accuracy))

    print('ACC decline: {}'.format(ori_accuracy - adv_accuracy))
    print('样本数： ' + str(adv_saved + ori_saved))
