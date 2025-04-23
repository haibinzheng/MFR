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
import sys
import shutil  # 用于复制文件

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets import SHIP_CLASSES
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

image_process = ValTransform(legacy=False)
img_test_size = (640, 640)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy"]
os.environ["CUDA_VISABLE_DEVICES"] = "1,2,3,4,5,6,7"
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import shutil
import subprocess

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

        if self.device != "cpu":
            # img = img.cuda()
            img = img.to(device)

            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            # 执行模型推理
            outputs = self.model(img)

            # print(self.decoder)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs_post, outputs_conf = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # 7维，1-4边界框坐标信息，5预测框内是否包含物体的置信度，或者说目标存在的置信度(概率)，6表示预测某一个类别的置信程度，7表示模型预测的目标所属类别。
            # 每个 bounding box 包含了85个数值，其中前四个数值是坐标信息，第五个数值是置信度，接下来的80个数值是每个类别的概率。
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, outputs_post, img_info, outputs_conf

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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True)  # 使用ResNet18作为基础模型
        self.fc = nn.Linear(1000, 2)  # 二分类任务，输出维度为2

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


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

    # image, _ = image_process(img, None, img_test_size)

    # image = torch.tensor(image)

    original_image_height = image.size[1]
    original_image_width = image.size[0]

    # 处理裁剪框超出范围的情况
    if box[0] < 0:
        box[0] = 0
    if box[1] < 0:
        box[1] = 0
    if box[2] > original_image_height:
        box[2] = original_image_height
    if box[3] > original_image_width:
        box[3] = original_image_width

    x1, y1, x2, y2 = box.tolist()
    a1, a2, b1, b2 = math.ceil(x1), math.ceil(y1), int(math.fabs(x2 - x1)), int(math.fabs(y2 - y1))

    cropped_image = image.crop([y1, x1, y2, x2])

    new_size = (80, 80)
    cro_image = cropped_image.resize(new_size, Image.ANTIALIAS)

    width, height = cro_image.size
    pixel_values_triplicated = conf * 3
    for i in range(3):
        for j, pixel_value in enumerate(pixel_values_triplicated[i * len(conf):(i + 1) * len(conf)]):
            cro_image.putpixel((width - 1 - len(conf) + j, height - 1),
                               (int(pixel_value * 255), int(pixel_value * 255), int(pixel_value * 255)))

    return cro_image


def yolox_detection(img_file, Method, detector, predictor):
    img = Image.open(img_file)
    outputs_adv, outputs_post_adv, img_info_adv, outputs_conf_adv = predictor.inference(img_file)

    if outputs_post_adv[0] is None:
        logger.info(f"outputs_post_adv is None")
        return 1

    box_idx_adv = None
    max_distance = 0
    max_box_idx = None

    # 遍历对抗样本中的所有框，找到最大框距离的框
    for idx, box_adv in enumerate(outputs_post_adv[0]):
        max_box_distance = max_boxDistance(box_adv[0:4])  # 计算当前框的“框距离”
        if max_box_distance > max_distance:
            max_distance = max_box_distance
            box_idx_adv = idx

    # 提取框信息
    box1 = outputs_post_adv[0][box_idx_adv][0:4]  # 最大框距离的框的坐标
    conf1 = outputs_conf_adv[0][box_idx_adv]  # 对应的置信度

    img = crop_and_save_image(img_file, box1, conf1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((80, 80)),
    ])

    transformed_image = transform(img).unsqueeze(0)


    return transformed_image

def detect_main(detparams):
    print(f"device={device}")

    exp = get_exp(None, 'yolox-x', )
    model = exp.get_model()
    model.to(device)

    ckpt_file = 'yolox_x.pth'

    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    model.eval()
    predictor = Predictor(
        model, exp, COCO_CLASSES, None, None,
        'gpu', False, False,
    )

    labels = []
    inputs = []
    total_time = 0
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    Attack_Method = detparams['Attack_Method']
    Method = detparams['Method']

    test_image_dir = '/data/Newdisk/chenjingwen/DT_B4/IndicatorsCoverage/SJS/PrePro_Plus/YOLOX/dataset/COCO/backdoor'

    # detector_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/yolo_detect/model_data/ALL_warship_v2_CNN_Classifier.pth'
    # detector = SimpleCNN()
    detector = CNN().to(device)
    # detector.load_state_dict(torch.load(detector_path))
    # detector.eval()
    if os.path.exists(test_image_dir):
        logger.info(f"开始检测！！！")
        for root, dirs, files in os.walk(test_image_dir):  # 以该数据集为例，可替换

            for file in files:
                if file.lower().endswith(supported_extensions):
                    file_path = os.path.join(root, file)
                    print(file_path)
                    # 打开图像
                    img = Image.open(file_path)

                    # 检查图像模式
                    if img.mode != 'RGB':
                        continue

                    transformed_image = yolox_detection(file_path, Method, detector, predictor)


                    if 'ori' in file_path.split('/'):
                        labels.append(1)
                        transformed_image = transformed_image.squeeze(0)

                        # 转换为 NumPy 数组并调整维度顺序 (C, H, W) -> (H, W, C)
                        transformed_image_np = transformed_image.permute(1, 2, 0).numpy()

                        # 转换为 PIL 图像并保存
                        img = Image.fromarray((transformed_image_np * 255).astype(np.uint8))
                        img.save('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/yolo_detect/temp/ori/{}.png'.format(os.path.splitext(file)[0]))


                    else:
                        labels.append(0)
                       
                        transformed_image = transformed_image.squeeze(0)

                        # 转换为 NumPy 数组并调整维度顺序 (C, H, W) -> (H, W, C)
                        transformed_image_np = transformed_image.permute(1, 2, 0).numpy()

                        # 转换为 PIL 图像并保存
                        img = Image.fromarray((transformed_image_np * 255).astype(np.uint8))
                        img.save('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/yolo_detect/temp/poi/{}.png'.format(os.path.splitext(file)[0]))

                    start_time = time.time()



                    end_time = time.time()

                    total_time += (end_time - start_time)
            

    # correct_predictions = sum(p == l for p, l in zip(results, labels))
    # accuracy = correct_predictions / len(labels)
    # avg_time = total_time / len(labels)
    # print("ACC: " + str(accuracy))
    # print("avg_time: " + str(avg_time))

    # 测试模型
    detector.eval()  # 设置模型为评估模式
    # with torch.no_grad():
    #     test_outputs = detector(inputs_tensor).squeeze()
    #     predictions = (test_outputs >= 0.5).float()  # 二分类阈值设为 0.5
    #     accuracy = (predictions == labels_tensor).sum().item() / len(labels_tensor)
    #     print(f'Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    detparams = {
        # 前后端传输
        'TaskId': '12378',
        'ModelDIR': 'yolox_x.pth',
        'Method': 'PrePro_CNNClassifier',
        'Attack_Method': 'ALL_warship_v2'

    }
    result = detect_main(detparams)
    print(result)
