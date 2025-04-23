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
from ceshi import SSD
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1")

###找别人在其他脚本中添加的print操作
# import sys
# import traceback

# old_f = sys.stdout

# class F1:
# def write(self, x):
# if "f_out0 shape before upsample" in x:
# stack_info = traceback.extract_stack()[-2]
# print(f"Found 'f_out0 shape before upsample' at file: {stack_info.filename}, line: {stack_info.lineno}")
# else:
# old_f.write(x)

# def flush(self):
# pass

# sys.stdout = F1()


# 追踪print打印位置
# old_f = sys.stdout
# class F1:
# def write(self, x):
# if x == "x1 shape: torch.Size([1, 640, 40, 40])":
# old_f.write(x.replace("\n", " [%s]\n" % str(traceback.extract_stack())))
# sys.stdout = F1()
# class F1:
# def write(self, x):
# if "x1 shape: torch.Size([1, 640, 40, 40])" in x:
# old_f.write(x.replace("\n", " [%s]\n" % str(traceback.extract_stack())))
# else:
# old_f.write(x)
# 图片处理
image_process = ValTransform(legacy=False)
img_test_size = (640, 640)


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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((80, 80)),
])


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
            # print(f"img_v1={img.shape}")
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 4:
                img = img[0]
            # print(f"img_v2={img.shape}")
        else:
            img_info["file_name"] = None

        height, width = img.shape[0:2]  # height, width = img.shape[0:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            # img = img.cuda()
            img = img.to(device)

            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            # print(f"the img={img.shape}")
            # print(f"the img={type(img)}")
            # print(f"img_v3={img.shape}")
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs_post = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # 7维，1-4边界框坐标信息，5预测框内是否包含物体的置信度，或者说目标存在的置信度(概率)，6表示预测某一个类别的置信程度，7表示模型预测的目标所属类别。
            # 每个 bounding box 包含了85个数值，其中前四个数值是坐标信息，第五个数值是置信度，接下来的80个数值是每个类别的概率。
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, outputs_post, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, [0]
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        # print(f"cls={cls}")
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


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
    IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy"]
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
def crop_and_save_image(image_path, box, conf, save_path):
    image = Image.open(image_path)
    conf = conf[:len(conf) - 1]
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
    new_size = (80, 80)
    cro_image = cropped_image.resize(new_size, Image.ANTIALIAS)

    new_list = [int(value * 255) for value in conf]

    print('conf', new_list)

    # print(f"ori_cro_image={np.array(cro_image)}")
    # print(f"ori_cro_image={np.array(cro_image).shape}")

    width, height = cro_image.size
    pixel_values_triplicated = conf * 3
    for i in range(3):
        for j, pixel_value in enumerate(pixel_values_triplicated[i * len(conf):(i + 1) * len(conf)]):
            cro_image.putpixel((width - 1 - len(conf) + j, height - 1),
                               (int(pixel_value * 255), int(pixel_value * 255), int(pixel_value * 255)))

    # print(f"new_cro_image={np.array(cro_image)}")
    # print(f"new_cro_image={np.array(cro_image).shape}")

    cropped_image.save(save_path)


def train_binary_classifier(params):
    ssd = SSD()
    # 建立目录
    # attack_methods = ['FGSM', 'IGSM', 'MIFGSM', 'PGD', 'patch_attack', 'GuassianBlurAttack', 'backdoor']
    attack_methods = params["Attack_Methods"]
    # for i in range(len(attack_methods)):
    #     os.makedirs(
    #         "../datasets/ssd_zz/Dataset_SSD/{}/train/adv".format(attack_methods[i]),
    #         exist_ok=True)
    #     os.makedirs(
    #         "../datasets/ssd_zz/Dataset_SSD/{}/test/adv".format(attack_methods[i]),
    #         exist_ok=True)
    #     os.makedirs(
    #         "../datasets/ssd_zz/Dataset_SSD/{}/train/ori".format(attack_methods[i]),
    #         exist_ok=True)
    #     os.makedirs(
    #         "../datasets/ssd_zz/Dataset_SSD/{}/test/ori".format(attack_methods[i]),
    #         exist_ok=True)
    #     os.makedirs(
    #         "../datasets/ssd_zz/Dataset_SSD/DetectData/{}/ori".format(attack_methods[i]),
    #         exist_ok=True)
    #     os.makedirs(
    #         "../datasets/ssd_zz/Dataset_SSD/DetectData/{}/adv".format(attack_methods[i]),
    #         exist_ok=True)

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
    # detect_path = []
    # detect_ground_label = []
    pickNumber_v1 = 800  # 2500 白盒攻击正常样本2000张，'FGSM','IGSM','MIFGSM','PGD'','patch_attack'各500张
    # pickNumber_v1 = 500 #黑盒攻击正常样本500张，GaussianBlurAttack也500张

    random.seed(7)
    ###读取对抗样本
    All_Attack_Methods = ['patch_attack']
    # All_Attack_Methods = ['backdoor']
    # All_Attack_Methods = ['GuassianBlurAttack']
    # All_Attack_Methods = ['backdoor']
    # All_Attack_Methods=['mosaic']

    for i in range(len(All_Attack_Methods)):
        advImage_number = 0
        # adv_path='/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true/{}'.format(All_Attack_Methods[i])
        if All_Attack_Methods[i] == 'backdoor':
            # adv_path = '/data0/BigPlatform/DT_Project/YOLOX/datasets/COCO/val2017/backdoor'
            adv_path = '../datasets/VOC_adv_image_true/backdoor_patch'
        else:
            # adv_path='/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true/{}'.format(All_Attack_Methods[i])
            adv_path = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/patch_result"

        pickNumber_v2 = pickNumber_v1 / len(All_Attack_Methods)
        print(f"pickNumber_v2={pickNumber_v2}")
        # pickNumber_v2=pickNumber_v1
        if os.path.isdir(adv_path):
            adv_file = get_image_list(adv_path)
        else:
            adv_file = [adv_path]
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
            if advImage_number < 600 + 1:
                train_path.append(image_name)
                train_ground_label.append(1)
                # cv2.imwrite(os.path.join(train_adv_path,new_image_basename),adv_image)
                # np.save(os.path.join(train_adv_path,new_image_basename),adv_image)
                number1 += 1
            if advImage_number > 600:
                test_path.append(image_name)
                test_ground_label.append(1)
                # cv2.imwrite(os.path.join(test_adv_path,new_image_basename),adv_image)
                # np.save(os.path.join(test_adv_path,new_image_basename),adv_image)
                number2 += 1
            if advImage_number > pickNumber_v2 - 1:
                print(f"读取{All_Attack_Methods[i]}的对抗样本结束！！！")
                # detect_path.append(image_name)
                # detect_ground_label.append(1)
                break
        adv_total_length = len(adv_file)
        print(f"adv_total_length_{All_Attack_Methods[i]}={adv_total_length}")
        print(f"number1_{All_Attack_Methods[i]}={number1}")
        print(f"number2_{All_Attack_Methods[i]}={number2}")

    ###读正常样本
    # ori_path = os.path.join('/data0/BigPlatform/DT_Project/001-Dataset/COCO/images/val2017')
    ori_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/SHIP/jpg'
    ### 自制正常数据集
    # ori_path = '/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh/ori'
    # ori_save_path = os.path.join('/data0/BigPlatform/DT_Project/001-Dataset/COCO/images')
    if os.path.isdir(ori_path):
        ori_file = get_image_list(ori_path)
    else:
        ori_file = [ori_path]

    ori_total_length = len(ori_file)
    print(f"ori_total_length={ori_total_length}")
    if pickNumber_v1 > ori_total_length:
        pickNumber_v1 = ori_total_length
        print(f"pickNumber_v1过大，超过了ori_total_length(5000)")

    number3 = 0
    number4 = 0
    # 保证ori_file中保存的图片和adv_file中的成对存在

    if attack_methods[0] == 'backdoor':
        for image_name in ori_file:
            image_files.append(image_name)
            oriImage_number += 1
            if oriImage_number < 600 + 1:
                train_path.append(image_name)
                train_ground_label.append(0)
                number3 += 1
            if oriImage_number > 600:
                test_path.append(image_name)
                test_ground_label.append(0)
                number4 += 1
            if oriImage_number > pickNumber_v1 - 1:
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
                if oriImage_number < 600 + 1:
                    train_path.append(image_name)
                    train_ground_label.append(0)
                    number3 += 1
                if oriImage_number > 600:
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
    # '''
    ### 生成自制训练数据集
    num = int(len(train_path) / 2)
    for i in range(num):

        imageOri_pil = Image.open(train_path[num + i])
        imageAdv_pil = Image.open(train_path[i])

        logger.info(f"num={i},imageOri_pil={imageOri_pil.size}")
        logger.info(f"num={i},imageAdv_pil={imageAdv_pil.size}")

        _, outputs_post_ori, outputs_conf_ori = ssd.detect_image(imageOri_pil)

        _, outputs_post_adv, outputs_conf_adv = ssd.detect_image(imageAdv_pil)

        closest_box_idx = None
        box_idx_ori = None
        box_idx_adv = None
        # closest_box_idx = []
        closest_distance = float('inf')
        max_distance = 0
        max_distance111 = 0
        max_box_idx = []
        if outputs_post_adv == '0' or outputs_post_ori == '0':
            logger.info(f"没检测到物体！！！")
            continue
        if attack_methods[0] == 'backdoor':
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
        conf1 = outputs_conf_ori[box_idx_ori]
        conf2 = outputs_conf_adv[box_idx_adv]
        # 图片裁剪和保存
        crop_and_save_image(train_path[num + i], box1, conf1, os.path.join(
            "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/plus_detectData/patch_attack/train/ori",
            os.path.basename(train_path[num + i])))
        crop_and_save_image(train_path[i], box2, conf2, os.path.join(
            "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/plus_detectData/patch_attack/train/adv",
            os.path.basename(train_path[i])))
        # exit()
    logger.info(f"训练样本数：{num}")
    time.sleep(10)
    # '''
    ### 源正常数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/COCO/images/val2017，源对抗数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true
    ### 自制数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh，预处理数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/test
    ### 生成自制测试数据集
    # '''
    ### 生成自制测试数据集
    num111 = int(len(test_path) / 2)
    for i in range(num111):

        imageOri_pil_test = Image.open(test_path[num111 + i])
        imageAdv_pil_test = Image.open(test_path[i])

        _, outputs_post_ori, outputs_conf_ori = ssd.detect_image(imageOri_pil_test)
        _, outputs_post_adv, outputs_conf_adv = ssd.detect_image(imageAdv_pil_test)

        if outputs_post_adv == '0' or outputs_post_ori == '0':
            logger.info(f"没检测到物体！！！")
            continue

        closest_box_idx = None
        box_idx_ori = None
        box_idx_adv = None
        closest_distance = float('inf')
        max_distance = 0
        max_box_idx = []
        max_distance111 = 0

        if attack_methods[0] == 'backdoor':
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
        conf1 = outputs_conf_ori[box_idx_ori]
        conf2 = outputs_conf_adv[box_idx_adv]
        # 图片裁剪和保存
        crop_and_save_image(test_path[num111 + i], box1, conf1, os.path.join(
            "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/plus_detectData/patch_attack/test/ori",
            os.path.basename(test_path[num111 + i])))
        crop_and_save_image(test_path[i], box2, conf2, os.path.join(
            "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/plus_detectData/patch_attack/test/adv",
            os.path.basename(test_path[i])))
    logger.info(f"测试样本数：{num111}")
    # '''

    ### 源正常数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/COCO/images/val2017，源对抗数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true
    ### 自制数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh，预处理数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/test

    path111 = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/plus_detectData/patch_attack/train"
    path222 = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/plus_detectData/patch_attack/test"
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

    # logger.info(f"****开始训练*******")
    #
    # # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #
    # # Initialize the dataset
    # dataset = datasets.ImageFolder(
    #     root='../datasets/ssd_zz/Dataset_SSD/{}/train'.format(attack_methods[0]),
    #     transform=transform)
    # test_dataset = datasets.ImageFolder(
    #     root='../datasets/ssd_zz/Dataset_SSD/{}/test'.format(attack_methods[0]),
    #     transform=transform)
    #
    # # Define batch size for DataLoader
    # batch_size = 10
    #
    # # Initialize the DataLoader
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #
    # # 模型训练
    # # model=MLP().to(device)
    # # model=CNNClassifier().to(device)
    # # model=linear_model(714000,2).to(device)
    # # model = SimpleCNN().to(device)
    # model = CNN().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # criterion = torch.nn.CrossEntropyLoss()
    # # criterion = torch.nn.MSELoss()
    # # criterion = torch.nn.BCELoss()
    # # criterion = torch.nn.BCEWithLogitsLoss()
    # logger.info(f"---training------")
    # train_epochs = 10
    # # for i in range(10):
    # # print(f"train_path[i]={train_path[i+1250]}")
    # # print(f"test_path[i]={test_path[i+1250]}")
    # # exit()
    # ###打乱训练集合测试集
    # # zipped_train_data = list(zip(train_path,train_ground_label))
    # # random.shuffle(zipped_train_data)
    # # shuffled_train_path, shuffled_train_ground_label = zip(*zipped_train_data)
    # # zipped_test_data = list(zip(test_path,test_ground_label))
    # # random.shuffle(zipped_test_data)
    # # shuffled_test_path, shuffled_test_ground_label = zip(*zipped_test_data)
    # Best_Acc = 0
    # saved_flag = False
    # saved_path = ''
    # for epoch in range(train_epochs):
    #     logger.info(f"---epoch={epoch}---")
    #     model.train()
    #     running_loss = 0
    #     train_correct = 0
    #     test_correct = 0
    #     total = 0
    #     total111 = 0
    #     for images, labels in data_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         predicted_train = torch.argmax(outputs.data, 1)
    #         train_correct += (predicted_train == labels).sum().item()
    #         # logger.info(f"predicted_train={predicted_train}={predicted_train.shape}")
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         total += labels.shape[0]
    #         running_loss += loss
    #     logger.info(
    #         f"epoch={epoch},Train_Accuarcy={train_correct}/{total}={train_correct / total},running_loss={running_loss}")
    #     with torch.no_grad():
    #         model.eval()
    #         for images111, labels111 in test_loader:
    #             images111, labels111 = images111.to(device), labels111.to(device)
    #             outputs111 = model(images111)
    #             predicted_test = torch.argmax(outputs111.data, 1)
    #             test_correct += (predicted_test == labels111).sum().item()
    #             total111 += images111.shape[0]
    #     Test_Accuarcy = test_correct / total111
    #     logger.info(f"epoch={epoch},Test_Accuarcy={test_correct}/{total111}={test_correct / total111}")
    #     if Test_Accuarcy > Best_Acc and Test_Accuarcy > 0.80:
    #         # print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
    #         Best_Acc = Test_Accuarcy
    #         logger.info(f"成功保存！！！")
    #         saved_flag = True
    #         torch.save(model.state_dict(),
    #                    '../datasets/ssd_zz/model/{}_CNN_Classifier_{}.pth'.format(
    #                        attack_methods[0], Test_Accuarcy))
    #         saved_path = '../datasets/ssd_zz/model/{}_CNN_Classifier_{}.pth'.format(
    #             attack_methods[0], Test_Accuarcy)
    # logger.info(f"All_Attack_Methods={All_Attack_Methods[0]}")
    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": attack_methods,
        'saved_model_path': saved_path if saved_flag else 'acc too low, not saved'
    }

    return jsontext


def CNN_Classifier_Detect(params):
    ssd = SSD()
    # 建立目录
    attack_methods = params["Attack_Methods"]
    # attack_methods = ['FGSM', 'IGSM', 'MIFGSM', 'PGD', 'patch_attack', 'GuassianBlurAttack', 'backdoor']
    for i in range(len(attack_methods)):
        os.makedirs(
            "../datasets/ssd_zz/Dataset_SSD/DetectData/{}/ori".format(attack_methods[i]),
            exist_ok=True)
        os.makedirs(
            "../datasets/ssd_zz/Dataset_SSD/DetectData/{}/adv".format(attack_methods[i]),
            exist_ok=True)

    image_files = []
    # 检测集
    detect_path = []
    detect_ground_label = []
    detect_result = []

    pickNumber = 300

    random.seed(7)
    ###读取对抗样本
    # All_Attack_Methods=['FGSM','IGSM','MIFGSM','PGD','patch_attack','GuassianBlurAttack','backdoor']
    for Attack_Method in attack_methods:
        # All_Attack_Methods = ['GuassianBlurAttack']
        # All_Attack_Methods = ['backdoor']
        advImage_number = 0
        # adv_path='/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true/{}'.format(All_Attack_Methods[i])
        if Attack_Method == 'backdoor':
            adv_path = '../datasets/VOC_adv_image_true/backdoor_patch'
        else:
            # adv_path='/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true/{}'.format(All_Attack_Methods[i])
            adv_path = "../datasets/VOC_adv_image_true/{}".format(Attack_Method)

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
        ori_path = os.path.join('../datasets/VOCdevkit/VOC2007/JPEGImages')
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

        ### 生成自制测试数据集
        count = 0
        num222 = int(len(detect_path) / 2)
        for i in range(num222):

            imageOri_pil_detect = Image.open(detect_path[num222 + i])
            imageAdv_pil_detect = Image.open(detect_path[i])

            _, outputs_post_ori, outputs_conf_ori = ssd.detect_image(imageOri_pil_detect)
            _, outputs_post_adv, outputs_conf_adv = ssd.detect_image(imageAdv_pil_detect)
            # print(outputs_post_ori)
            # print(outputs_conf_ori)
            # print('原始box是:', outputs_post_ori)
            # print('原始置信度是:',outputs_conf_ori)

            logger.info(f"detect_path[num222+i]={detect_path[num222 + i]}")
            logger.info(f"detect_path[i]={detect_path[i]}")
            if outputs_post_adv == '0' or outputs_post_ori == '0':
                logger.info(f"没检测到物体！！！")
                continue
            count += 1
            if count > 200:
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
            conf1 = outputs_conf_ori[box_idx_ori]

            box2 = outputs_post_adv[box_idx_adv]
            conf2 = outputs_conf_adv[box_idx_adv]

            logger.info(f"box1={box1}")
            # 图片裁剪和保存
            crop_and_save_image(detect_path[num222 + i], box1, conf1, os.path.join(
                "../datasets/ssd_zz/Dataset_SSD/DetectData/{}/ori".format(Attack_Method),
                os.path.basename(detect_path[num222 + i])))
            print('kkkkkkkkkkkkkkkkkkk的i值', i)
            crop_and_save_image(detect_path[i], box2, conf2, os.path.join(
                "../datasets/ssd_zz/Dataset_SSD/DetectData/{}/adv".format(Attack_Method),
                os.path.basename(detect_path[i])))
        logger.info(f"测试样本数：{num222}")

        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # Initialize the dataset
        test_dataset = datasets.ImageFolder(
            root='../datasets/ssd_zz/Dataset_SSD/DetectData/{}'.format(Attack_Method),
            transform=transform)

        # Define batch size for DataLoader
        batch_size = 10

        # Initialize the DataLoader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        model = CNN().to(device)
        if Attack_Method == 'FGSM':
            model_path = '../datasets/ssd_zz/model/FGSM_CNN_Classifier_0.9929577464788732.pth'
        elif Attack_Method == 'IGSM':
            model_path = '../datasets/ssd_zz/model/IGSM_CNN_Classifier_0.9931506849315068.pth'
        elif Attack_Method == 'patch_attack':
            model_path = '../datasets/ssd_zz/model/patch_attack_CNN_Classifier_0.9418604651162791.pth'
        elif Attack_Method == 'GuassianBlurAttack':
            model_path = '../datasets/ssd_zz/model/GuassianBlurAttack_CNN_Classifier_0.9707602339181286.pth'
        elif Attack_Method == 'PGD':
            model_path = '../datasets/ssd_zz/model/PGD_CNN_Classifier_0.9438202247191011.pth'
        elif Attack_Method == 'MIFGSM':
            model_path = '../datasets/ssd_zz/model/MIFGSM_CNN_Classifier_1.0.pth'
        elif Attack_Method == 'backdoor':
            model_path = '../datasets/ssd_zz/model/backdoor_CNN_Classifier_0.9031413612565445.pth'
        # model_path = '../datasets/ssd_zz/model/FGSM_CNN_Classifier_0.9929577464788732.pth'
        # model_path = '../datasets/ssd_zz/model/IGSM_CNN_Classifier_0.9931506849315068.pth'
        # model_path = '../datasets/ssd_zz/model/patch_attack_CNN_Classifier_0.9418604651162791.pth'
        # model_path = '../datasets/ssd_zz/model/GuassianBlurAttack_CNN_Classifier_0.9707602339181286.pth'
        # model_path = '../datasets/ssd_zz/model/backdoor_CNN_Classifier_0.9031413612565445.pth'
        model.load_state_dict(torch.load(model_path))

        test_correct = 0
        total111 = 0
        model.eval()
        print(f"test_loader={len(test_loader)}")
        path333 = "../datasets/ssd_zz/Dataset_SSD/DetectData/{}".format(Attack_Method)
        if os.path.isdir(path333):
            file333 = get_image_list(path333)
        else:
            file333 = [path333]
        print(f"file333={len(file333)}")
        # test_loader=random.sample(list(test_loader),200)
        start_time = time.time()
        for images111, labels111 in test_loader:
            images111, labels111 = images111.to(device), labels111.to(device)
            outputs111 = model(images111)
            predicted_test = torch.argmax(outputs111.data, 1)
            for item in predicted_test.cpu().numpy():
                detect_result.append(item)
            test_correct += (predicted_test == labels111).sum().item()
            total111 += images111.shape[0]
            # if total111 > 199:
            # break
        end_time = time.time()
        everage_time = (end_time - start_time) / len(file333)
        Test_Accuarcy = test_correct / total111
        logger.info(f"CNN_Classifier_Detect:Test_Accuarcy={test_correct}/{total111}={test_correct / total111}")
        logger.info(f"everage_time={everage_time},Attack_Method={Attack_Method}")
    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": attack_methods,
        'average_time': everage_time,
        "detect_result": detect_result
    }

    return jsontext


def detect_main(params):
    ssd = SSD()
    if params['mode'] == 'train':
        result = train_binary_classifier(params)
        return result
    elif params['mode'] == 'test':
        result = CNN_Classifier_Detect(params)
        return result


if __name__ == "__main__":
    params = {
        "taskId": 12345,
        "mode": 'train',  # {train、test}
        # "Model": '/data0/BigPlatform/DT_B4/GD_detect/object_detection/Yolo_detect/save_model/FGSM_CNN_Classifier_0.9803921568627451.pth',
        "Attack_Methods": ["ALL"]  # {FGSM、IGSM、GuassianBlurAttack、PSO_patch
    }
    result = detect_main(params)
    print(result)
