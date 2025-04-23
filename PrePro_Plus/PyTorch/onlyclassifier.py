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
from sdd_zz import SSD
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)

os.environ["CUDA_VISABLE_DEVICES"] = "1,2,3,4,5,6,7"
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((80, 80)),
])
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

    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    return distance


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
def crop_and_save_image(image_path, box, conf, save_path):
    # image1 = Image.open(image_path)
    # img = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR) #PIL转为opencv
    img = cv2.imread(image_path)
    image, _ = image_process(img, None, img_test_size)
    # print(f"image={image.shape}={type(image)}")
    image = torch.tensor(image)
    # image = TF.to_tensor(image)
    # image = cv2.imread(image_path)
    # tensor_image = TF.to_tensor(image)
    original_image_height = image.shape[1]
    original_image_width = image.shape[2]
    # logger.info(f"image.shape={image.shape}")
    # logger.info(f"image={type(image)}")
    # logger.info(f"image_path={image_path}")
    # logger.info(f"original_image_width={original_image_width}")
    # logger.info(f"original_image_height={original_image_height}")

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
    # print(f"x1={int(x1)},y1={int(y1)},x2={int(x2)},y2={int(y2)}")
    # print(f"a1={a1},a2={a2},b1=x2-x1={b1},b2=y2-y1={b2}")
    # cropped_box = (max(0, box[0]), max(0, box[1]), min(image_width, box[2]), min(image_height, box[3]))
    # cropped_image = TF.crop(image, math.ceil(y1), math.ceil(x1), math.ceil(y2-y1), math.ceil(x2-x1))
    cropped_image = TF.crop(image, a2, a1, b2, b1)
    cropped_image = tensor_to_image(cropped_image)
    cropped_image = Image.fromarray(cropped_image)
    # print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
    # print(type(cropped_image))
    new_size = (80, 80)
    cro_image = cropped_image.resize(new_size, Image.ANTIALIAS)
    # 转化为0-255
    # new_list = [int(value * 255) for value in conf]
    # print('pppppppppppppppppppppppppppppppp')
    # print(cro_image.size)
    # print('conf', new_list)
    # print(len(new_list))
    # exit()
    # print(f"ori_cro_image={np.array(cro_image)}")
    # print(f"ori_cro_image={np.array(cro_image).shape}")
    # exit(0)
    width, height = cro_image.size
    pixel_values_triplicated = conf * 3
    for i in range(3):
        for j, pixel_value in enumerate(pixel_values_triplicated[i * len(conf):(i + 1) * len(conf)]):
            cro_image.putpixel((width - 1 - len(conf) + j, height - 1),
                               (int(pixel_value * 255), int(pixel_value * 255), int(pixel_value * 255)))
    # logger.info(f"cropped_image={cropped_image.shape}={type(cropped_image)}")
    # cropped_image = TF.to_pil_image(cropped_image)
    # logger.info(f"类型转换！！！")
    # cropped_image.save(save_path)
    cro_image.save(save_path)
    # logger.info(f"-----裁剪完成------")


def train_binary_classifier(params):
    # 建立目录
    # attack_methods = ['FGSM', 'IGSM', 'MIFGSM', 'PGD', 'PSO_patch', 'GuassianBlurAttack', 'backdoor']
    IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy"]
    all_attack_methods = params["Attack_Methods"]

    logger.info(f"****开始训练*******")

    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # Initialize the dataset
    dataset = datasets.ImageFolder(
            root='/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/plus_detectData/ALL',
        transform=transform)
    test_dataset = datasets.ImageFolder(
        root='/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/warship_ssd/plus_detectData/ALL',
        transform=transform)

    # Define batch size for DataLoader
    batch_size = 1

    # Initialize the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 模型训练
    # model=MLP().to(device)
    # model=CNNClassifier().to(device)
    # model=linear_model(714000,2).to(device)
    # model = SimpleCNN().to(device)
    model = CNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.BCELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    logger.info(f"---training------")
    train_epochs = 20
    # for i in range(10):
    # print(f"train_path[i]={train_path[i+1250]}")
    # print(f"test_path[i]={test_path[i+1250]}")
    # exit()
    ###打乱训练集合测试集
    # zipped_train_data = list(zip(train_path,train_ground_label))
    # random.shuffle(zipped_train_data)
    # shuffled_train_path, shuffled_train_ground_label = zip(*zipped_train_data)
    # zipped_test_data = list(zip(test_path,test_ground_label))
    # random.shuffle(zipped_test_data)
    # shuffled_test_path, shuffled_test_ground_label = zip(*zipped_test_data)
    # model.load_state_dict(torch.load('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/ssd_detect/model_data/ALL_v2_VOC_CNN_Classifier_1.0.pth'))
    Best_Acc = 0
    saved_flag = False
    saved_path = ''
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
        if Test_Accuarcy > Best_Acc and Test_Accuarcy > 0.50:
            Best_Acc = Test_Accuarcy
            logger.info(f"成功保存！！！")
            saved_flag = True
            torch.save(model.state_dict(),
                       '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/ssd_detect/model_data/ALL_warship_CNN_Classifier_{}.pth'.format(
                           Test_Accuarcy))
            saved_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/ssd_detect/model_data/ALL_warship_CNN_Classifier_{}.pth'.format(
                           Test_Accuarcy)

    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": all_attack_methods,
        'saved_model_path': saved_path if saved_flag else 'acc too low, not saved'
    }

    return jsontext


def CNN_Classifier_Detect(params):
    # train_binary_classifier()
    # all_attack_methods = params["Attack_Methods"]
    # all_attack_methods=['FGSM','IGSM','MIFGSM','PGD','PSO_patch','GuassianBlurAttack','backdoor']
    all_attack_methods = ['FGSM','patch_attack']
    detect_result = []
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    for attack_method in all_attack_methods:
        output_dir = f'/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/ssd_detect/select_warship/{attack_method}'
        # Initialize the dataset
        dataset = datasets.ImageFolder(
            root=f'/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/ssd_detect/select_warship/{attack_method}',
            transform=transform)
        test_dataset = datasets.ImageFolder(
            root=f'/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/ssd_detect/select_warship/{attack_method}',
            transform=transform)

        # Define batch size for DataLoader
        batch_size = 1

        # Initialize the DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # model = binary_model(2).to(device)
        model = CNN().to(device)
        model_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/ssd_detect/model_data/ALL_warship_CNN_Classifier_0.7393483709273183.pth'
        # if attack_method == 'FGSM':
        #     model_path = 'model_data/FGSM_CNN_Classifier_0.9026548672566371.pth'
        # elif attack_method == 'IGSM':
        #     model_path = 'model_data/IGSM_CNN_Classifier_0.8724489795918368.pth'
        # elif attack_method == 'MIFGSM':
        #     model_path = 'model_data/MIFGSM_CNN_Classifier_0.8814432989690721.pth'
        # elif attack_method == 'PGD':
        #     model_path = 'model_data/PGD_CNN_Classifier_0.6011904761904762.pth'
        # elif attack_method == 'PSO_patch':
        #     model_path = 'model_data/PSO_patch_CNN_Classifier_0.5972222222222222.pth'
        # elif attack_method == 'GuassianBlurAttack':
        #     model_path = 'model_data/GuassianBlurAttack_CNN_Classifier_0.6838235294117647.pth'
        # elif attack_method == 'backdoor':
        #     model_path = '../datasets/yolo_zz/backdoor_CNN_Classifier_0.555.pth'

        model.load_state_dict(torch.load(model_path))

        test_correct = 0
        total111 = 0
        model.eval()
        print(f"test_loader={len(test_loader)}")
        path333 = f'/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/ssd_detect/select_warship/{attack_method}'
        if os.path.isdir(path333):
            file111 = get_image_list(path333)
        else:
            file111 = [path333]
        print(f"file111={len(file111)}")
        # test_loader=random.sample(list(test_loader),200)
        start_time = time.time()
        iii = 0
        i = 0
        ii = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # print(target)
            outputs111 = model(data)

            predicted_test = torch.argmax(outputs111.data, 1)
            for item in predicted_test.cpu().numpy():

                # if target[0] == 1:
                #     if item == target[0]:
                #         # print('检测成功！')
                #         i += 1
                #         input1 = data.squeeze(0)
                #         # print(input1)
                #         input1 = input1.permute(1, 2, 0)  # 将(C, H, W)转为(H, W, C)
                #         input1 = torch.clamp(input1, 0, 1)  # 将值限制在[0, 1]之间
                #         array = input1.cpu().numpy()
                #         array = (array * 255).astype(np.uint8)
                #         img = Image.fromarray(array)
                #         os.makedirs(output_dir + f'/ori', exist_ok=True)
                #         img.save(os.path.join(output_dir + f'/ori/img_{i}.jpg'))
                #         iii += 1
                #     else:
                #         if random.random() < 0.05:
                #             i += 1
                #             input1 = data.squeeze(0)
                #             # print(input1)
                #             input1 = input1.permute(1, 2, 0)  # 将(C, H, W)转为(H, W, C)
                #             input1 = torch.clamp(input1, 0, 1)  # 将值限制在[0, 1]之间
                #             array = input1.cpu().numpy()
                #             array = (array * 255).astype(np.uint8)
                #             img = Image.fromarray(array)
                #             os.makedirs(output_dir + f'/ori', exist_ok=True)
                #             img.save(os.path.join(output_dir + f'/ori/img_{i}.jpg'))
                #             iii += 1
                # elif target[0] == 0:
                #     if item == target[0]:
                #         # print('检测成功！')
                #         ii += 1
                #         input1 = data.squeeze(0)
                #         # print(input1)
                #         input1 = input1.permute(1, 2, 0)  # 将(C, H, W)转为(H, W, C)
                #         input1 = torch.clamp(input1, 0, 1)  # 将值限制在[0, 1]之间
                #         array = input1.cpu().numpy()
                #         array = (array * 255).astype(np.uint8)
                #         img = Image.fromarray(array)
                #         os.makedirs(output_dir + f'/adv', exist_ok=True)
                #         img.save(os.path.join(output_dir + f'/adv/img_{ii}.jpg'))
                #         iii += 1
                #
                #     else:
                #         if random.random() < 0.05:
                #             ii += 1
                #             input1 = data.squeeze(0)
                #             # print(input1)
                #             input1 = input1.permute(1, 2, 0)  # 将(C, H, W)转为(H, W, C)
                #             input1 = torch.clamp(input1, 0, 1)  # 将值限制在[0, 1]之间
                #             array = input1.cpu().numpy()
                #             array = (array * 255).astype(np.uint8)
                #             img = Image.fromarray(array)
                #             os.makedirs(output_dir + f'/adv', exist_ok=True)
                #             img.save(os.path.join(output_dir + f'/adv/img_{ii}.jpg'))
                #             iii += 1

                detect_result.append(item)
            test_correct += (predicted_test == target).sum().item()
            total111 += data.shape[0]
            # if total111 > 199:
            # break
        end_time = time.time()
        everage_time = (end_time - start_time) / total111
        Test_Accuarcy = test_correct / total111
        logger.info(f"CNN_Classifier_Detect:Test_Accuarcy={test_correct}/{total111}={test_correct / total111}")
        logger.info(f"everage_time={everage_time},All_Attack_Methods={attack_method}")

    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": all_attack_methods,
        'average_time': everage_time,
        "detect_result": detect_result
    }

    return jsontext


def detect_main(params):

    if params['mode'] == 'train':
        result = train_binary_classifier(params)
        return result
    elif params['mode'] == 'test':
        result = CNN_Classifier_Detect(params)
        return result


if __name__ == "__main__":
    params = {
        "taskId": 12345,
        "mode": 'test',  # {train、test}
        # "Model": '/data0/BigPlatform/DT_B4/GD_detect/object_detection/Yolo_detect/save_model/FGSM_CNN_Classifier_0.9803921568627451.pth',
        "Attack_Methods": ["ALL_Attack"]  # {FGSM、IGSM、GuassianBlurAttack、PSO_patch
    }
    result = detect_main(params)
    print(result)
