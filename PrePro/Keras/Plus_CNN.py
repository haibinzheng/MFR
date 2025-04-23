import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from tqdm import tqdm
from yolo import YOLO
import random
from utils.utils import cvtColor, preprocess_input, resize_image
import math
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import logging
from loguru import logger
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_test_size = (640, 640)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        print(e)


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


def image_process(img, res, input_size):
    # 你的image_process函数实现
    # 此处为示例实现，具体细节需要根据你的实际情况调整
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32)
    img = img / 255.0
    img -= np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    img /= np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = np.transpose(img, (2, 0, 1))  # swap channels
    return img, np.zeros((1, 5))


# 图片裁剪和保存
def crop_and_save_image_2(image_path, box, conf, save_path):
   
    # 读取图像
    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 设置输入图像大小，假设为 (640, 640)
    input_size = (640, 640)
    img, _ = image_process(img, None, input_size)

    # 将图像转换为TensorFlow张量
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # 获取图像的高度和宽度
    original_image_height, original_image_width = img_tensor.shape[1], img_tensor.shape[2]

    # 处理裁剪框超出范围的情况
    box = tf.convert_to_tensor(box, dtype=tf.int32)
    box = tf.clip_by_value(box, [0, 0, 0, 0],
                           [original_image_width, original_image_height, original_image_width, original_image_height])

    x1, y1, x2, y2 = box.numpy().tolist()

    # 裁剪图像
    cropped_image = img_tensor[:, y1:y2, x1:x2]

    new_size = (len(conf), len(conf))
    # cropped_image.resize(new_size)
    # 将conf的值提取并重复三次
    conf_values = conf.numpy()
    pixel_values_triplicated = tf.tile(conf, [3]).numpy()

    cropped_height, cropped_width, _ = cropped_image.shape
    cropped_image=cropped_image.numpy()
    # 将像素值添加到图像的最后一行
    for i in range(3):
        for j, pixel_value in enumerate(pixel_values_triplicated[i * len(conf_values):(i + 1) * len(conf_values)]):
            if cropped_width - 1 - len(conf_values) + j >= 0:
                cropped_image[cropped_height - 1, cropped_width - 1 - len(conf_values) + j, i] = int(pixel_value * 255)

    # 将裁剪后的图像转换为numpy数组并保存

    cropped_image = tf.transpose(cropped_image, perm=[1, 2, 0])  # 交换回HWC
    cropped_image = cropped_image.numpy()
    cropped_image = (cropped_image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # 反归一化
    cropped_image = (cropped_image * 255.0).astype(np.uint8)

    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)  # 转换回BGR格式

    cv2.imwrite(save_path, cropped_image)
    print("cccccccccc")
    return 0


def crop_and_save_image(image_path, box, conf, save_path):
    # 读取图像
    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 设置输入图像大小，假设为 (640, 640)
    input_size = (640, 640)
    img, _ = image_process(img, None, input_size)

    # 将图像转换为TensorFlow张量
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # 获取图像的高度和宽度
    original_image_height, original_image_width = img_tensor.shape[1], img_tensor.shape[2]

    # 处理裁剪框超出范围的情况
    box = tf.convert_to_tensor(box, dtype=tf.int32)
    box = tf.clip_by_value(box, [0, 0, 0, 0],
                           [original_image_width, original_image_height, original_image_width, original_image_height])

    x1, y1, x2, y2 = box.numpy().tolist()

    # 裁剪图像
    cropped_image = img_tensor[:, y1:y2, x1:x2]

    # 将裁剪后的图像转换为numpy数组并保存
    try:
        cropped_image = tf.transpose(cropped_image, perm=[1, 2, 0])  # 交换回HWC
        cropped_image = cropped_image.numpy()
        cropped_image = (cropped_image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # 反归一化
        cropped_image = (cropped_image * 255.0).astype(np.uint8)

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)  # 转换回BGR格式

        cv2.imwrite(save_path, cropped_image)
        return 0
    except:
        return 1


# 计算框之间的距离
def box_distance(box1, box2):
    center1 = get_center(box1)
    center2 = get_center(box2)

    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    return distance


def CNN():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # 不冻结基础模型的卷积层

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    return model


# 找到最大的一个框
def max_boxDistance(box):
    x1, y1, x2, y2 = box
    distance = int(math.fabs((x2 - x1) * (y2 - y1)))
    return distance


def train_binary_classifier(params):
    attack_methods = ['FGSM']
    # for i in range(len(attack_methods)):
    #     os.makedirs("dataset/{}/train/adv".format(attack_methods[i]), exist_ok=True)
    #     os.makedirs("dataset/{}/test/adv".format(attack_methods[i]), exist_ok=True)
    #     os.makedirs("dataset/{}/train/ori".format(attack_methods[i]), exist_ok=True)
    #     os.makedirs("dataset/{}/test/ori".format(attack_methods[i]), exist_ok=True)
    #     os.makedirs("dataset/DetectData/ori", exist_ok=True)
    #     os.makedirs("dataset/DetectData/adv", exist_ok=True)

    image_files = []
    oriImage_number = 0
    ground_label = []
    # 训练集
    train_path = []
    train_ground_label = []
    # 测试集
    test_path = []
    test_ground_label = []
    pickNumber_v1 = 800  # 2500 白盒攻击正常样本2000张，'FGSM','IGSM','MIFGSM','PGD'','PSO_patch'各500张
    # pickNumber_v1 = 500 #黑盒攻击正常样本500张，GaussianBlurAttack也500张

    random.seed(7)
    ###读取对抗样本
    All_Attack_Methods = params['Attack_Methods']

    for i in range(len(All_Attack_Methods)):
        advImage_number = 0
        # adv_path='/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true/{}'.format(All_Attack_Methods[i])
        if All_Attack_Methods[i] == 'backdoor':
            adv_path = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/COCO/val2017/backdoor'
        elif All_Attack_Methods[i] in ['PGD', 'backdoor']:
            adv_path = "/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection/datasets/YOLOX_adv_image_true/{}/test".format(
                All_Attack_Methods[i])
        else:
            adv_path='/data/Newdisk/chenjingwen/DT_B4/GD_detect/object_detection/datasets/YOLOX_adv_image_true/{}/test'.format(All_Attack_Methods[i])


        print(adv_path)
        ### 自制对抗数据集路径
        # adv_path='/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh/adv'

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
                break
        adv_total_length = len(adv_file)
        print(f"adv_total_length_{All_Attack_Methods[i]}={adv_total_length}")
        print(f"number1_{All_Attack_Methods[i]}={number1}")
        print(f"number2_{All_Attack_Methods[i]}={number2}")

    ori_path = os.path.join('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/object_detection/datasets/COCO/val2017/ori')
    # ori_path = os.path.join('/data0/BigPlatform/DT_Project/ssd-pytorch-master/VOCdevkit/VOC2007/JPEGImages')
    ### 自制正常数据集
    # ori_path = '/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh/ori'
    # ori_save_path = os.path.join('/data0/BigPlatform/DT_Project/001-Dataset/COCO/images')
    if os.path.isdir(ori_path):
        ori_file = get_image_list(ori_path)
    else:
        ori_file = [ori_path]

    ori_total_length = len(ori_file)
    print(f"ori_total_length={ori_total_length}")
    # if pickNumber_v1>ori_total_length:
    #     pickNumber_v1=ori_total_length
    #     print(f"pickNumber_v1过大，超过了ori_total_length(5000)")
    # train_ori_path=os.path.join(ori_save_path,'train','ori')
    # test_ori_path=os.path.join(ori_save_path,'test','ori')
    # os.makedirs(train_ori_path,exist_ok=True)
    # os.makedirs(test_ori_path,exist_ok=True)
    number3 = 0
    number4 = 0
    for i in range(len(adv_file)):
        # 保证ori_file中保存的图片和adv_file中的成对存在
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
                break
    print(f"number3={number3}")
    print(f"number4={number4}")
    print(f"train_ground_label={len(train_ground_label)},test_ground_label={len(test_ground_label)}")
    print(f"train_path={len(train_path)},test_path={len(test_path)}")
    print(f"读取样本后的总图片数量={len(image_files)}")
    time.sleep(10)

    yolo = YOLO(confidence=0.001, nms_iou=0.5)
    ### 生成自制训练数据集
    num = int(len(train_path) / 2)
    for i in range(num):
        image_1 = Image.open(train_path[i])
        image_2 = Image.open(train_path[num + i])

        outputs_post_adv, out_scores_adv = yolo.get_pred_3(image_1)
        outputs_post_ori, out_scores_ori = yolo.get_pred_3(image_2)

        if outputs_post_adv[0][0] is None or outputs_post_ori[0][0] is None:
            logger.info(f"outputs_post_adv is None or outputs_post_ori is None")
            continue

        # img_adv = predictor.visual(outputs_post_adv[0],a img_info_adv, cls_conf=0.35)
        # img_ori = predictor.visual(outputs_post_ori[0],img_info_ori,cls_conf=0.35)

        # cv2.imwrite(os.path.join("/data0/BigPlatform/DT_Project/002-Code/002-Detect/test/adv",os.path.basename(train_path[i])),img_adv)
        # cv2.imwrite(os.path.join("/data0/BigPlatform/DT_Project/002-Code/002-Detect/test/ori",os.path.basename(train_path[num+i])),img_ori)

        closest_box_idx = None
        box_idx_ori = None
        box_idx_adv = None
        # closest_box_idx = []
        closest_distance = float('inf')
        max_distance = 0
        max_box_idx = []
        # 找到outputs_post_adv中每一个框与outputs_post_ori中的最接近的框的idx
        # max_box_idx中存储的为ori中框的索引，max_box_idx的下标就是adv中框的索引

        outputs_post_adv = outputs_post_adv
        outputs_post_ori = outputs_post_ori
        for box_adv in outputs_post_adv:  # 每个框都保存，然后选框最大的一个
            # print(box_adv[0:4])
            for idx, box_ori in enumerate(outputs_post_ori):
                distance = box_distance(box_ori[0:4], box_adv[0:4])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_box_idx = idx
            max_box_idx.append(closest_box_idx)
            # print(max_box_idx)
        # 选框最大的一个框
       
        for j in range(len(max_box_idx)):
            max_box_distance = max_boxDistance(outputs_post_ori[max_box_idx[j]][0:4])
            if max_box_distance > max_distance:
                max_distance = max_box_distance
                box_idx_ori = max_box_idx[j]
                box_idx_adv = j
        box1 = outputs_post_ori[box_idx_ori][0:4]
        conf1 = out_scores_ori

        box2 = outputs_post_adv[box_idx_adv][0:4]
        conf2 = out_scores_adv

        # print(f"box1={box1},box2={box2},max_distance={max_distance}")
        # print(f"outputs_post_ori[0][box_idx_ori]={outputs_post_ori[0][box_idx_ori]},outputs_post_adv[0][box_idx_adv]={outputs_post_adv[0][box_idx_adv]}")
        # print(f"----------box_idx_ori={box_idx_ori},box_idx_adv={box_idx_adv}-----------")
        # 图片裁剪和保存
        crop_and_save_image(train_path[num + i], box1, conf1,
                              os.path.join("dataset/{}/train/ori".format(All_Attack_Methods[0]),
                                           os.path.basename(train_path[num + i])))
        crop_and_save_image(train_path[i], box2, conf2,
                              os.path.join("dataset/{}/train/adv".format(All_Attack_Methods[0]),
                                           os.path.basename(train_path[i])))
    logger.info(f"训练样本数：{num}")
    time.sleep(10)

    num111 = int(len(test_path) / 2)
    for i in range(num111):

        image_1 = Image.open(train_path[i])
        image_2 = Image.open(train_path[num + i])

        outputs_post_adv, out_scores_adv = yolo.get_pred_3(image_1)
        outputs_post_ori, out_scores_ori = yolo.get_pred_3(image_2)

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
        # closest_box_idx = []
        closest_distance = float('inf')
        max_distance = 0
        max_box_idx = []
        # 找到outputs_post_adv中每一个框与outputs_post_ori中的最接近的框的idx
        # max_box_idx中存储的为ori中框的索引，max_box_idx的下标就是adv中框的索引

        outputs_post_adv = outputs_post_adv
        outputs_post_ori = outputs_post_ori
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
        conf1 = out_scores_ori

        box2 = outputs_post_adv[box_idx_adv][0:4]
        conf2 = out_scores_adv
        # print(f"box1={box1},box2={box2},max_distance={max_distance}")
        # print(f"outputs_post_ori[0][box_idx_ori]={outputs_post_ori[0][box_idx_ori]},outputs_post_adv[0][box_idx_adv]={outputs_post_adv[0][box_idx_adv]}")
        # print(f"----------box_idx_ori={box_idx_ori},box_idx_adv={box_idx_adv}-----------")
        # 图片裁剪和保存

        result1 = crop_and_save_image(test_path[num111 + i], box1, conf1,
                                        os.path.join("dataset/{}/test/ori".format(All_Attack_Methods[0]),
                                                     os.path.basename(test_path[num111 + i])))
        if result1 == 0:
            result2 = crop_and_save_image(test_path[i], box2, conf2,
                                            os.path.join("dataset/{}/test/adv".format(All_Attack_Methods[0]),
                                                         os.path.basename(test_path[i])))
            if result2 == 1:
                num111 -= 1
        else:
            num111 -= 1

    logger.info(f"测试样本数：{num111}")

    jsontext = {
        'Message': '训练样本和测试样本已生成完毕',

    }

    return jsontext

    # '''
    ### 源正常数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/COCO/images/val2017，源对抗数据集路径：/data0/BigPlatform/DT_Project/001-Dataset/YOLOX_adv_image_true
    ### 自制数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/Dataset_zh，预处理数据集保存路径：/data0/BigPlatform/DT_Project/002-Code/002-Detect/test


def test(params):
    All_Attack_Methods = params['Attack_Methods'][0]
    # 定义路径
    path_train = 'dataset/{}/train'.format(All_Attack_Methods)
    path_test = 'dataset/{}/test'.format(All_Attack_Methods)

    path_train_ori = os.path.join(path_train, 'ori')
    path_train_adv = os.path.join(path_train, 'adv')
    path_test_ori = os.path.join(path_test, 'ori')
    path_test_adv = os.path.join(path_test, 'adv')

    # 获取图像列表函数
    # def get_image_list(directory):
    #     return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 检查路径并获取文件列表
    file_train_ori = get_image_list(path_train_ori) if os.path.isdir(path_train_ori) else []
    file_train_adv = get_image_list(path_train_adv) if os.path.isdir(path_train_adv) else []
    file_test_ori = get_image_list(path_test_ori) if os.path.isdir(path_test_ori) else []
    file_test_adv = get_image_list(path_test_adv) if os.path.isdir(path_test_adv) else []

    logger.info(
        f"train_ori_num={len(file_train_ori)}, train_adv_num={len(file_train_adv)}, test_ori_num={len(file_test_ori)}, test_adv_num={len(file_test_adv)}")
    time.sleep(10)

    logger.info(f"****开始训练*******")

    batch_size = 32

    # 预处理图像的函数
    def preprocess_image(image):

        image = image.resize((224, 224))
        image = np.array(image) / 255.0  # 归一化处理

        return image

    data_augmentation = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    # 创建数据集函数
    def create_dataset(file_list, label):
        images = []
        labels = []
        for file in file_list:
            image = Image.open(file).convert('RGB')
            image = preprocess_image(image)
            images.append(image)
            labels.append(label)
        return np.array(images), np.array(labels)

    # 创建训练集和测试集
    train_images_ori, train_labels_ori = create_dataset(file_train_ori, 0)
    train_images_adv, train_labels_adv = create_dataset(file_train_adv, 1)
    test_images_ori, test_labels_ori = create_dataset(file_test_ori, 0)
    test_images_adv, test_labels_adv = create_dataset(file_test_adv, 1)

    train_images = np.concatenate((train_images_ori, train_images_adv), axis=0)
    train_labels = np.concatenate((train_labels_ori, train_labels_adv), axis=0)
    test_images = np.concatenate((test_images_ori, test_images_adv), axis=0)
    test_labels = np.concatenate((test_labels_ori, test_labels_adv), axis=0)

    # 创建 TensorFlow 数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_labels)).batch(
        batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    model = CNN()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005, decay=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_epochs = 15
    Best_Acc = 0

    for epoch in range(train_epochs):
        logger.info(f"---epoch={epoch}---")

        model.fit(train_dataset, epochs=1, verbose=1, shuffle=True)

        # 评估训练集准确率
        train_loss, train_acc = model.evaluate(train_dataset, verbose=1)
        print(model(train_images))
        logger.info(f"epoch={epoch}, Train_Accuracy={train_acc}, running_loss={train_loss}")

        # 评估测试集准确率
        test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
        print(f"epoch={epoch}, Test_Accuracy={test_acc}, running_loss={test_loss}")

        # 保存模型
        if test_acc > Best_Acc:
            Best_Acc = test_acc
            model.save(f"save_model/{All_Attack_Methods}_CNN_acc_{test_acc:.4f}.h5")

    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": All_Attack_Methods,
        'Save_path': 'save_model/{}_CNN_Classifier_{}.h5'.format(All_Attack_Methods[0], Test_Accuarcy)
    }

    return jsontext


def detect_main(params):
    if params['mode'] == 'train':
        result = train_binary_classifier(params)
        return result
    elif params['mode'] == 'test':
        result = train_binary_classifier(params)
        return result


if __name__ == "__main__":
    IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy"]
    params = {
        "taskId": 12345,
        "mode": 'train',  # {train、test}
        "Attack_Methods": ["FGSM"]  # {FGSM、IGSM、GuassianBlurAttack、PSO_patch
    }
    result = detect_main(params)
    print(result)
