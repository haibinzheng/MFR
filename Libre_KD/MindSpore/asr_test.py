import time
import datetime
import glob
import sys

import os
import numpy as np
import torch.nn as nn
import torch
from mindspore import Tensor, context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
# from mindspore.ops import functional as F
import torch.nn.functional as F
from mindspore.common import dtype as mstype
import mindspore.dataset.vision as vision
from src.utils.logging import get_logger
from src.vgg_1 import vgg16_sentiment, Vgg_sentiment
from src.dataset import vgg_create_dataset
from src.dataset import classification_dataset
from src.dataset import create_dataset

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num

from model_utils.config import get_config
from PIL import Image
import pickle
import shutil
import random

im_width, im_height = 224, 224
mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
transform_img = [
    vision.Decode(),
    vision.Resize((256, 256)),
    vision.CenterCrop((224, 224)),
    vision.Normalize(mean=mean, std=std),
    vision.HWC2CHW()
]


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 更新为 128 * 1 * 1
        self.fc1 = nn.Linear(128 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 2)  # 二分类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 1 * 1)  # 展平

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def model_load(model_path, type):
    if (type == 'classify'):
        model = vgg16_sentiment(config.num_classes, config, phase="test")
        load_param_into_net(model, load_checkpoint(model_path))
        model.add_flags_recursive(fp16=True)
        model.set_train(False)

    if (type == 'detect'):
        model = SimpleCNN()
        model.load_state_dict(
            torch.load(model_path))
        model.eval()

    return model


def vgg_fgsm_detection(image, model):
    detection_model = model[0]
    classification_model = model[1]

    img = image.read()
    for op in transform_img:
        img = op(img)
    img = np.expand_dims(img, axis=0)

    feature = []
    # x = Tensor(img, mstype.float32)
    #
    # for i in range(len(classification_model.layers)):
    #     x = classification_model.layers[i](x)
    #     if i == 5 or i == 14 or i == 35:
    #         feature.append(x.asnumpy())
    #
    # input_tensor = torch.tensor(feature[-1], dtype=torch.float32)
    #
    # output = detection_model(input_tensor)
    # predicted_label = torch.argmax(output).item()

    output2 = classification_model(Tensor(img, mstype.float32))
    class_pred = np.argmax(output2)

    return predicted_label, class_pred


if __name__ == "__main__":
    FGSM_IMAGES_PATH = "/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/selected_2/military/VGG16/PatchAttack"  # 测试数据集路径

    VGG16_MODEL_PATH = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image_mindspore/save_models/0-55_153_military.ckpt'
    FGSM_MODEL_PATH = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image_mindspore/detectors/detector_Military_PatchAttack.pth"  # 测试检测模型路径

    model = []
    FGSM_MODEL = model_load(FGSM_MODEL_PATH, "detect")
    VGG16_MODEL = model_load(VGG16_MODEL_PATH, "classify")
    model.append(FGSM_MODEL)
    model.append(VGG16_MODEL)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    results = []
    labels = []
    ori_correct = 0
    adv_correct = 0
    ori_total = 0
    adv_total = 0

    total_time = 0
    all_files = []  # 用于存储所有文件的路径

    # 遍历文件夹，收集所有文件路径
    for root, dirs, files in os.walk(FGSM_IMAGES_PATH):  # 可替换为你的数据集路径
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)
                all_files.append((file_path, root))  # 保存文件路径及其所在目录

    # 打乱文件顺序
    random.shuffle(all_files)
    print('总文件数目： {}'.format(len(all_files)))

    # 按打乱的顺序处理文件
    for file_path, root in all_files:
        class_name = int(os.path.basename(root))  # 假设文件夹名是类别

        # 判断是 'ori' 还是 'adv' 图片
        if 'ori' in file_path.split('/'):
            labels.append(0)
            ori_total += 1
        else:
            labels.append(1)
            adv_total += 1

        with open(file_path, "rb") as img_file:
            start_time = time.time()
            result, class_pred = vgg_fgsm_detection(img_file, model)
            end_time = time.time()
            results.append(result)
            total_time += (end_time - start_time)

            if 'ori' in file_path.split('/') and class_pred == class_name:
                ori_correct += 1
            if 'adv' in file_path.split('/') and class_pred == class_name:
                adv_correct += 1

print("MindSpore PatchAttack ORI ACC: " + str(ori_correct / ori_total))

print("MindSpore PatchAttack ADV ACC: " + str(adv_correct / adv_total))

print('ACC Decline: {}'.format(ori_correct / ori_total - adv_correct / adv_total))
