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

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from utils.utils import cvtColor, preprocess_input, resize_image
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

image_process = ValTransform(legacy=False)
img_test_size = (640, 640)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy"]
os.environ["CUDA_VISABLE_DEVICES"] = "1,2,3,4,5,6,7"
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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


def crop_and_save_image(image_path, box, conf):
    # image1 = Image.open(image_path)
    # img = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR) #PIL转为opencv
    img = cv2.imread(image_path)
    image, _ = image_process(img, None, img_test_size)

    image = torch.tensor(image)

    original_image_height = image.shape[1]
    original_image_width = image.shape[2]



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

    cropped_image = TF.crop(image, a2, a1, b2, b1)
    cropped_image = tensor_to_image(cropped_image)
    cropped_image = Image.fromarray(cropped_image)

    new_size = (80, 80)
    cro_image = cropped_image.resize(new_size, Image.ANTIALIAS)
    #
    # width, height = cro_image.size
    # pixel_values_triplicated = conf * 3
    # for i in range(3):
    #     for j, pixel_value in enumerate(pixel_values_triplicated[i * len(conf):(i + 1) * len(conf)]):
    #         cro_image.putpixel((width - 1 - len(conf) + j, height - 1),
    #                            (int(pixel_value * 255), int(pixel_value * 255), int(pixel_value * 255)))

    return cro_image
    # logger.info(f"-----裁剪完成------")


def yolox_detection(img_file, model):
   
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
    output = model(transformed_image)
    predicted_test = torch.argmax(output.data, 1)


    return predicted_test.numpy()[0]


if __name__ == "__main__":


    FGSM_IMAGES_PATH = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/YOLOX_adv_image_true/select/FGSM"  # 测试数据集路径
    # DETECTOR_PATH = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection/Yolo_detect/save_model/ALL_COCO_CNN_Classifier_0.91735918744229.pth'
    DETECTOR_PATH = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/yolo_detect/model_data/ALL_Attack_CNN_Classifier_0.7896440129449838.pth'
    exp = get_exp(None, 'yolox-x', )
    model = exp.get_model()
    model.to(device)

    ckpt_file = 'yolox_x.pth'

    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    model.eval()

    logger.info("loaded checkpoint done.")
    
    detector = model_load(DETECTOR_PATH)
    predictor = Predictor(
        model, exp, COCO_CLASSES, None, None,
        'gpu', False, False,
    )

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    results = []
    labels = []
    total_time = 0
    for root, dirs, files in os.walk(FGSM_IMAGES_PATH):  # 以该数据集为例，可替换

        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)

                if 'ori' in file_path.split('/'):
                    labels.append(0)
                else:
                    labels.append(1)

                start_time = time.time()

                result = yolox_detection(file_path, detector)
                end_time = time.time()
                results.append(result)
                print(file_path)
                # if 'ori' in file_path.split('/') and result ==0:
                #     shutil.copy(file_path,  os.path.join('/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection/Yolo_detect/select_COCO/FGSM/ori', os.path.basename(file)))
                # elif'ori' in file_path.split('/') and result == 1:
                #     if random.random() < 0.05:
                #         shutil.copy(file_path, os.path.join(
                #             '/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection/Yolo_detect/select_COCO/FGSM/ori',
                #             os.path.basename(file)))
                #     print("良性判断错了")
                # if 'adv' in file_path.split('/') and result == 1:
                #     shutil.copy(file_path,  os.path.join('/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection/Yolo_detect/select_COCO/FGSM/adv', os.path.basename(file)))
                # elif 'adv' in file_path.split('/') and result == 0:
                #     if random.random() < 0.05:
                #         shutil.copy(file_path, os.path.join(
                #             '/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection/Yolo_detect/select_COCO/FGSM/adv',
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
