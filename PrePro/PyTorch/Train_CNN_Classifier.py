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
from utils.utils import cvtColor, preprocess_input, resize_image
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
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import random
import time
import torchvision.transforms.functional as TF

import torchvision.models as models

os.environ["CUDA_VISABLE_DEVICES"] = "1,2,3,4,5,6,7"
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:1")

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
            #print(f"img_v1={img.shape}")
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 4:
                img = img[0]
            #print(f"img_v2={img.shape}")
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
            #img = img.cuda()
            img = img.to(device)
            
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            # print(f"the img={img.shape}")
            # print(f"the img={type(img)}")
            #print(f"img_v3={img.shape}")
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs_post = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            #7维，1-4边界框坐标信息，5预测框内是否包含物体的置信度，或者说目标存在的置信度(概率)，6表示预测某一个类别的置信程度，7表示模型预测的目标所属类别。
            #每个 bounding box 包含了85个数值，其中前四个数值是坐标信息，第五个数值是置信度，接下来的80个数值是每个类别的概率。
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
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
        #print(f"cls={cls}")
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res
class linear_model(torch.nn.Module):
    def __init__(self,input_size,output_size):
        super(linear_model,self).__init__()
        self.linear1 = torch.nn.Linear(input_size,512)
        self.act1=torch.nn.ReLU(True)
        self.linear2 = torch.nn.Linear(512, 512)
        self.act2 = torch.nn.ReLU(True)
        self.linear3 = torch.nn.Linear(512, 512)
        self.act3 = torch.nn.ReLU(True)
        self.linear4 = torch.nn.Linear(512, output_size)
#        self.act4 = torch.nn.ReLU(True)
#        self.linear5 = torch.nn.Linear(1024, output_size)
        self.act5 = torch.nn.Softmax(dim=1)
        # self.act5 = torch.nn.Sigmoid()



    def forward(self,x):
        x=self.linear1(x)
        x=self.act1(x)
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
        self.fc1 = nn.Linear(8*357000, 256)  # 根据输入形状计算得到
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
        x = x.view(-1, 8*357000)  # 根据池化层的输出形状计算得到
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
    x1,y1,x2,y2 = box
    distance = int(math.fabs((x2-x1) * (y2-y1)))
    return distance
# 图片裁剪和保存
def crop_and_save_image(image_path, box, save_path):
    # image1 = Image.open(image_path)
    # img = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR) #PIL转为opencv
    img = cv2.imread(image_path)
    image, _ = image_process(img, None, img_test_size)
    #print(f"image={image.shape}={type(image)}")
    image = torch.tensor(image)
    #image = TF.to_tensor(image)
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
    a1,a2,b1,b2 = math.ceil(x1),math.ceil(y1),int(math.fabs(x2-x1)),int(math.fabs(y2-y1))
    # print(f"x1={int(x1)},y1={int(y1)},x2={int(x2)},y2={int(y2)}")
    # print(f"a1={a1},a2={a2},b1=x2-x1={b1},b2=y2-y1={b2}")
    #cropped_box = (max(0, box[0]), max(0, box[1]), min(image_width, box[2]), min(image_height, box[3]))
    #cropped_image = TF.crop(image, math.ceil(y1), math.ceil(x1), math.ceil(y2-y1), math.ceil(x2-x1))
    cropped_image = TF.crop(image, a2, a1, b2, b1)
    #logger.info(f"cropped_image={cropped_image.shape}={type(cropped_image)}")
    # cropped_image = TF.to_pil_image(cropped_image)
    # logger.info(f"类型转换！！！")
    # cropped_image.save(save_path)
    cv2.imwrite(save_path,cropped_image.permute(1,2,0).numpy())
    #logger.info(f"-----裁剪完成------")        

def train_binary_classifier(params,Predictor):
    predictor = Predictor
    # 建立目录
    attack_methods=['MIFGSM']
    for i in range(len(attack_methods)):
        os.makedirs("dataset/{}/train/adv".format(attack_methods[i]),exist_ok=True)
        os.makedirs("dataset/{}/test/adv".format(attack_methods[i]),exist_ok=True)
        os.makedirs("dataset/{}/train/ori".format(attack_methods[i]),exist_ok=True)
        os.makedirs("dataset/{}/test/ori".format(attack_methods[i]),exist_ok=True)
        os.makedirs("dataset/DetectData/ori",exist_ok=True)
        os.makedirs("dataset/DetectData/adv",exist_ok=True)
    
    image_files=[]
    oriImage_number=0        
    ground_label=[]
    # 训练集
    train_path = []
    train_ground_label = []
    # 测试集
    test_path = []
    test_ground_label = []
    pickNumber_v1=800 #2500 白盒攻击正常样本2000张，'FGSM','IGSM','MIFGSM','PGD'','PSO_patch'各500张
    #pickNumber_v1 = 500 #黑盒攻击正常样本500张，GaussianBlurAttack也500张

    random.seed(7)
   ###读取对抗样本
    All_Attack_Methods = params['Attack_Methods']
  
    for i in range(len(All_Attack_Methods)):
        advImage_number=0
        #adv_path='/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true/{}'.format(All_Attack_Methods[i])
        if All_Attack_Methods[i] == 'backdoor':
            adv_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/COCO/val2017/backdoor'
        elif All_Attack_Methods[i] in ['PGD','backdoor']:
            adv_path = "/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection/datasets/YOLOX_adv_image_true/{}/test".format(
                All_Attack_Methods[i])
        else:
            #adv_path='/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true/{}'.format(All_Attack_Methods[i])
            adv_path = "/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection/datasets/YOLOX_adv_image_true/{}/test".format(All_Attack_Methods[i])

        print(adv_path)
        ### 自制对抗数据集路径
        #adv_path='/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh/adv'
        
        pickNumber_v2=pickNumber_v1/len(All_Attack_Methods)
        print(f"pickNumber_v2={pickNumber_v2}")
        #pickNumber_v2=pickNumber_v1
        if os.path.isdir(adv_path):
            adv_file = get_image_list(adv_path)
        else:
            adv_file = [adv_path]
        number1=0
        number2=0
        random.shuffle(adv_file)
        for image_name in adv_file:
            image_files.append(image_name)
            ground_label.append(1)
            advImage_number+=1
            # adv_image = cv2.imread(image_name)
            #adv_image = np.load(image_name)
            # image_basename = os.path.basename(image_name)
            # new_image_basename = image_basename[:-4]+All_Attack_Methods[i]+'.jpg'
            if advImage_number < 600+1:
                train_path.append(image_name)
                train_ground_label.append(1)
                #cv2.imwrite(os.path.join(train_adv_path,new_image_basename),adv_image)
                #np.save(os.path.join(train_adv_path,new_image_basename),adv_image)
                number1+=1
            if advImage_number > 600:
                test_path.append(image_name)
                test_ground_label.append(1)
                #cv2.imwrite(os.path.join(test_adv_path,new_image_basename),adv_image)
                #np.save(os.path.join(test_adv_path,new_image_basename),adv_image)
                number2+=1
            if advImage_number>pickNumber_v2-1:
                print(f"读取{All_Attack_Methods[i]}的对抗样本结束！！！")
                break
        adv_total_length=len(adv_file)
        print(f"adv_total_length_{All_Attack_Methods[i]}={adv_total_length}")
        print(f"number1_{All_Attack_Methods[i]}={number1}")
        print(f"number2_{All_Attack_Methods[i]}={number2}")
    
   ###读正常样本
    ori_path = os.path.join('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/COCO/val2017/ori')
    # ori_path = os.path.join('/data0/BigPlatform/DT_Project/ssd-pytorch-master/VOCdevkit/VOC2007/JPEGImages')
    ### 自制正常数据集
    #ori_path = '/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh/ori'
    #ori_save_path = os.path.join('/data0/BigPlatform/DT_Project/001-Dataset/COCO/images')
    if os.path.isdir(ori_path):
        ori_file = get_image_list(ori_path)
    else:
        ori_file = [ori_path]

    ori_total_length=len(ori_file)
    print(f"ori_total_length={ori_total_length}")
    # if pickNumber_v1>ori_total_length:
    #     pickNumber_v1=ori_total_length
    #     print(f"pickNumber_v1过大，超过了ori_total_length(5000)")
    # train_ori_path=os.path.join(ori_save_path,'train','ori')
    # test_ori_path=os.path.join(ori_save_path,'test','ori')
    # os.makedirs(train_ori_path,exist_ok=True)
    # os.makedirs(test_ori_path,exist_ok=True)
    number3=0
    number4=0
    for i in range(len(adv_file)):
        # 保证ori_file中保存的图片和adv_file中的成对存在
        base_name = os.path.basename(adv_file[i])
        image_name = os.path.join(ori_path,base_name)
        if image_name in ori_file:
            image_files.append(image_name)        
            oriImage_number+=1
            if oriImage_number < 600+1:
                train_path.append(image_name)
                train_ground_label.append(0)
                number3+=1
            if oriImage_number > 600:
                test_path.append(image_name)
                test_ground_label.append(0)                
                number4+=1
            if oriImage_number>pickNumber_v1-1:
                print(f"原始数据读取完毕！！！")
                break   
    print(f"number3={number3}")
    print(f"number4={number4}")
    print(f"train_ground_label={len(train_ground_label)},test_ground_label={len(test_ground_label)}")
    print(f"train_path={len(train_path)},test_path={len(test_path)}")
    print(f"读取样本后的总图片数量={len(image_files)}")
    time.sleep(10)
    
### 生成自制训练数据集        
    #'''       
    ### 生成自制训练数据集
    num = int(len(train_path)/2)
    for i in range(num):           
        outputs_adv,outputs_post_adv,img_info_adv = predictor.inference(train_path[i])
        outputs_ori,outputs_post_ori,img_info_ori = predictor.inference(train_path[num+i])
        # print(outputs_post_ori[0])
        
        
        if outputs_post_adv[0][0] is None or outputs_post_ori[0][0] is None:
            logger.info(f"outputs_post_adv is None or outputs_post_ori is None")
            continue

        # img_adv = predictor.visual(outputs_post_adv[0], img_info_adv, cls_conf=0.35)
        # img_ori = predictor.visual(outputs_post_ori[0],img_info_ori,cls_conf=0.35)
        
        # cv2.imwrite(os.path.join("/data0/BigPlatform/DT_Project/002-Code/002-Detect/test/adv",os.path.basename(train_path[i])),img_adv)
        # cv2.imwrite(os.path.join("/data0/BigPlatform/DT_Project/002-Code/002-Detect/test/ori",os.path.basename(train_path[num+i])),img_ori)
                                
        closest_box_idx = None
        box_idx_ori = None
        box_idx_adv = None
        #closest_box_idx = []
        closest_distance = float('inf')
        max_distance = 0
        max_box_idx = []
        # 找到outputs_post_adv中每一个框与outputs_post_ori中的最接近的框的idx
        # max_box_idx中存储的为ori中框的索引，max_box_idx的下标就是adv中框的索引
        for box_adv in outputs_post_adv[0][0]: # 每个框都保存，然后选框最大的一个
            # print(box_adv[0:4])
            for idx,box_ori in enumerate(outputs_post_ori[0][0]):
                distance = box_distance(box_ori[0:4], box_adv[0:4])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_box_idx = idx
            max_box_idx.append(closest_box_idx)
            # print(max_box_idx)
        # 选框最大的一个框
        for j in range(len(max_box_idx)):
            max_box_distance = max_boxDistance(outputs_post_ori[0][0][max_box_idx[j]][0:4])
            if max_box_distance > max_distance:
                max_distance = max_box_distance
                box_idx_ori = max_box_idx[j]
                box_idx_adv = j
        box1 = outputs_post_ori[0][0][box_idx_ori][0:4]
        box2 = outputs_post_adv[0][0][box_idx_adv][0:4]
        # print(f"box1={box1},box2={box2},max_distance={max_distance}")
        # print(f"outputs_post_ori[0][box_idx_ori]={outputs_post_ori[0][box_idx_ori]},outputs_post_adv[0][box_idx_adv]={outputs_post_adv[0][box_idx_adv]}")
        # print(f"----------box_idx_ori={box_idx_ori},box_idx_adv={box_idx_adv}-----------")
        # 图片裁剪和保存
        crop_and_save_image(train_path[num+i], box1, os.path.join("dataset/{}/train/ori".format(All_Attack_Methods[0]),os.path.basename(train_path[num+i])))
        crop_and_save_image(train_path[i], box2, os.path.join("dataset/{}/train/adv".format(All_Attack_Methods[0]),os.path.basename(train_path[i])))
    logger.info(f"训练样本数：{num}")       
    time.sleep(10)
    #'''           
### 源正常数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/COCO/images/val2017，源对抗数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true
### 自制数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh，预处理数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/test        
### 生成自制测试数据集        
    #'''       
    ### 生成自制测试数据集
    num111 = int(len(test_path)/2)
    for i in range(num111):           
        outputs_adv,outputs_post_adv,img_info_adv = predictor.inference(test_path[i])
        outputs_ori,outputs_post_ori,img_info_ori = predictor.inference(test_path[num111+i])
        
        # print(f"test_path[i]={test_path[i]}")
        # #print(f"outputs_post_adv={outputs_post_adv}")
        # print(f"test_path[num111+i]={test_path[num111+i]}")
        # #print(f"outputs_post_ori={outputs_post_ori}")
        if outputs_post_adv[0][0] is None or outputs_post_ori[0][0] is None:
            logger.info(f"outputs_post_adv is None or outputs_post_ori is None")
            continue

        # img_adv = predictor.visual(outputs_post_adv[0], img_info_adv, cls_conf=0.35)
        # img_ori = predictor.visual(outputs_post_ori[0],img_info_ori,cls_conf=0.35)
        
        # cv2.imwrite(os.path.join("/data0/BigPlatform/DT_Project/002-Code/002-Detect/test/adv",os.path.basename(test_path[i])),img_adv)
        # cv2.imwrite(os.path.join("/data0/BigPlatform/DT_Project/002-Code/002-Detect/test/ori",os.path.basename(test_path[num111+i])),img_ori)
                                
        closest_box_idx = None
        box_idx_ori = None
        box_idx_adv = None
        #closest_box_idx = []
        closest_distance = float('inf')
        max_distance = 0
        max_box_idx = []
        # 找到outputs_post_adv中每一个框与outputs_post_ori中的最接近的框的idx
        # max_box_idx中存储的为ori中框的索引，max_box_idx的下标就是adv中框的索引
        for box_adv in outputs_post_adv[0][0]: # 每个框都保存，然后选框最大的一个
            for idx,box_ori in enumerate(outputs_post_ori[0][0]):
                distance = box_distance(box_ori[0:4], box_adv[0:4])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_box_idx = idx
            max_box_idx.append(closest_box_idx)            
        # 选框最大的一个框
        for j in range(len(max_box_idx)):
            max_box_distance = max_boxDistance(outputs_post_ori[0][0][max_box_idx[j]][0:4])
            if max_box_distance > max_distance:
                max_distance = max_box_distance
                box_idx_ori = max_box_idx[j]
                box_idx_adv = j
        box1 = outputs_post_ori[0][0][box_idx_ori][0:4]
        box2 = outputs_post_adv[0][0][box_idx_adv][0:4]
        # print(f"box1={box1},box2={box2},max_distance={max_distance}")
        # print(f"outputs_post_ori[0][box_idx_ori]={outputs_post_ori[0][box_idx_ori]},outputs_post_adv[0][box_idx_adv]={outputs_post_adv[0][box_idx_adv]}")
        # print(f"----------box_idx_ori={box_idx_ori},box_idx_adv={box_idx_adv}-----------")
        # 图片裁剪和保存
        crop_and_save_image(test_path[num111+i], box1, os.path.join("dataset/{}/test/ori".format(All_Attack_Methods[0]),os.path.basename(test_path[num111+i])))
        crop_and_save_image(test_path[i], box2, os.path.join("dataset/{}/test/adv".format(All_Attack_Methods[0]),os.path.basename(test_path[i])))
    logger.info(f"测试样本数：{num111}")   
    #'''           
### 源正常数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/COCO/images/val2017，源对抗数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true
### 自制数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh，预处理数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/test        
    
    path111 = "dataset/{}/train".format(All_Attack_Methods[0])
    path222 = "dataset/{}/test".format(All_Attack_Methods[0])
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
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((224, 224)),
    # ])
    #     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #
    # # Initialize the dataset
    # dataset = datasets.ImageFolder(root='dataset/{}/train'.format(All_Attack_Methods[0]), transform=transform)
    # test_dataset = datasets.ImageFolder(root='dataset/{}/test'.format(All_Attack_Methods[0]), transform=transform)
    #
    # # Define batch size for DataLoader
    # batch_size = 10
    #
    # # Initialize the DataLoader
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #
    # # 模型训练
    # #model=MLP().to(device)
    # #model=CNNClassifier().to(device)
    # #model=linear_model(714000,2).to(device)
    # #model = SimpleCNN().to(device)
    # model = CNN().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)
    # #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # criterion = torch.nn.CrossEntropyLoss()
    # #criterion = torch.nn.MSELoss()
    # #criterion = torch.nn.BCELoss()
    # #criterion = torch.nn.BCEWithLogitsLoss()
    # logger.info(f"---training------")
    # train_epochs = 15
    # # for i in range(10):
    #     # print(f"train_path[i]={train_path[i+1250]}")
    #     # print(f"test_path[i]={test_path[i+1250]}")
    # # exit()
    # ###打乱训练集合测试集
    # # zipped_train_data = list(zip(train_path,train_ground_label))
    # # random.shuffle(zipped_train_data)
    # # shuffled_train_path, shuffled_train_ground_label = zip(*zipped_train_data)
    # # zipped_test_data = list(zip(test_path,test_ground_label))
    # # random.shuffle(zipped_test_data)
    # # shuffled_test_path, shuffled_test_ground_label = zip(*zipped_test_data)
    # Best_Acc = 0
    # for epoch in range(train_epochs):
    #     logger.info(f"---epoch={epoch}---")
    #     model.train()
    #     running_loss = 0
    #     train_correct = 0
    #     test_correct = 0
    #     total = 0
    #     total111 = 0
    #     for images,labels in data_loader:
    #         images,labels = images.to(device),labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         predicted_train = torch.argmax(outputs.data,1)
    #         train_correct +=(predicted_train == labels).sum().item()
    #         #logger.info(f"predicted_train={predicted_train}={predicted_train.shape}")
    #         loss = criterion(outputs,labels)
    #         loss.backward()
    #         optimizer.step()
    #         total += labels.shape[0]
    #         running_loss += loss
    #     logger.info(f"epoch={epoch},Train_Accuarcy={train_correct}/{total}={train_correct/total},running_loss={running_loss}")
    #     with torch.no_grad():
    #         model.eval()
    #         for images111,labels111 in test_loader:
    #             images111,labels111 = images111.to(device),labels111.to(device)
    #             outputs111 = model(images111)
    #             predicted_test = torch.argmax(outputs111.data,1)
    #             test_correct += (predicted_test == labels111).sum().item()
    #             total111 += images111.shape[0]
    #     Test_Accuarcy = test_correct/total111
    #     logger.info(f"epoch={epoch},Test_Accuarcy={test_correct}/{total111}={test_correct/total111}")
    #     if Test_Accuarcy > Best_Acc and Test_Accuarcy > 0.50:
    #         Best_Acc = Test_Accuarcy
    #         logger.info(f"成功保存！！！")
    #         torch.save(model.state_dict(),'save_model/{}_CNN_Classifier_{}.pth'.format(All_Attack_Methods[0],Test_Accuarcy))

    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": All_Attack_Methods,
        'Save_path':'save_model/{}_CNN_Classifier_{}.pth'.format(All_Attack_Methods[0],Test_Accuarcy)
    }

    return jsontext
    
         

def CNN_Classifier_Detect(params):
    IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy"]
    
    #All_Attack_Methods=['FGSM','IGSM','MIFGSM','PGD','PSO_patch','GuassianBlurAttack','backdoor']
    #All_Attack_Methods=['backdoor']
    All_Attack_Methods = params['Attack_Methods']
    detect_result=[]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # Initialize the dataset
    dataset = datasets.ImageFolder(root='dataset/{}/train'.format(All_Attack_Methods[0]), transform=transform)
    test_dataset = datasets.ImageFolder(root='dataset/{}/test'.format(All_Attack_Methods[0]), transform=transform)

    # Define batch size for DataLoader
    batch_size = 10

    # Initialize the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    model = CNN().to(device)
    if All_Attack_Methods[0]=='FGSM':
        model_path='save_model/FGSM_CNN_Classifier_0.9803921568627451.pth'
    elif All_Attack_Methods[0]=='IGSM':
        model_path='save_model/IGSM_CNN_Classifier_0.987012987012987.pth'
    elif All_Attack_Methods[0] == 'PGD':
        model_path = 'save_model/PGD_CNN_Classifier_0.7880434782608695.pth'
    elif All_Attack_Methods[0] == 'PSO_patch':
        model_path = 'save_model/PSO_patch_CNN_Classifier_0.8125.pth'
    elif All_Attack_Methods[0] == 'GaussianBlurAttack':
        model_path = 'save_model/GuassianBlurAttack_CNN_Classifier_0.7764705882352941.pth'
    elif All_Attack_Methods[0] == 'backdoor':
        model_path = 'save_model/backdoor_CNN_Classifier_0.54.pth'

    model.load_state_dict(torch.load(model_path))

    test_correct = 0
    total111 = 0
    model.eval()
    print(f"test_loader={len(test_loader)}")
    path333 = "dataset/{}/test".format(All_Attack_Methods[0])
    if os.path.isdir(path333):
        file111 = get_image_list(path333)
    else:
        file111 = [path333]
    print(f"file111={len(file111)}")
    #test_loader=random.sample(list(test_loader),200)
    start_time = time.time()
    for images111,labels111 in test_loader:
        images111,labels111 = images111.to(device),labels111.to(device)
        outputs111 = model(images111)
        predicted_test = torch.argmax(outputs111.data,1)
        for item in predicted_test.cpu().numpy():
            detect_result.append(item)
        test_correct += (predicted_test == labels111).sum().item()
        total111 += images111.shape[0]
        # if total111 > 199:
            # break
    end_time = time.time()
    everage_time = (end_time-start_time)/len(file111)
    Test_Accuarcy = test_correct/total111
    logger.info(f"CNN_Classifier_Detect:Test_Accuarcy={test_correct}/{total111}={test_correct/total111}") 
    logger.info(f"everage_time={everage_time},All_Attack_Methods={All_Attack_Methods[0]}")


    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": All_Attack_Methods[0],
        'average_time':everage_time,
        'detect_result':detect_result
    }

    return jsontext

def detect_main(params):
    IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy"]
    print(f"device={device}")
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name, )
    model = exp.get_model()
    model.to(device)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    if args.trt:
        args.device = "gpu"
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    args.c = 'yolox_x.pth'
    if params['Attack_Methods'][0]=='backdoor':
        args.c = 'best_ckpt.pth'
    # args.c='/data0/BigPlatform/DT_Project/YOLOX/YOLOX_outputs/yolox_x/best_ckpt.pth'

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.c
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    if params['mode'] == 'train':
        result = train_binary_classifier(params,predictor)
        return result
    elif params['mode'] == 'test':
        result = CNN_Classifier_Detect(params)
        return result








if __name__ == "__main__":
    IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png",".npy"]
    params = {
        "taskId": 12345,
        "mode": 'train', #{train、test}
       
        "Attack_Methods": ["MIFGSM"]  #{FGSM、IGSM、GuassianBlurAttack、PSO_patch
    }
    result = detect_main(params)
    print(result)

    
