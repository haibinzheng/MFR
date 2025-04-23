import os
import time
import datetime
import glob
import numpy as np
import mindspore.nn as nn
import gc
from mindspore import Tensor, context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from nets.src.utils.logging import get_logger
from nets.src.vgg import vgg16, Vgg
from nets.src.dataset import vgg_create_dataset
from nets.src.dataset import classification_dataset
from nets.src.dataset import create_dataset
from sklearn.neighbors import KernelDensity
from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id, get_device_num
from PIL import Image
from model_utils.config import get_config
from util_kd import *
import time
import csv
from joblib import dump, load

BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00, 'imagenet': 0.15, 'military': 0.45}
im_width, im_height = 224, 224


def get_kd(Model, model, X_train, Y_train, X_test, X_test_adv, dataset_name):
    dataset = dataset_name
    kdes_path = 'kd_save/save_kdes/kdes_{}_{}.joblib'.format(dataset, Model)
    if os.path.isfile(kdes_path):
        print("kdes已经存在")
        kdes = load(kdes_path)
    else:
        print("kdes路径不存在，重新训练一个")

        if dataset_name == 'imagenet':
            Y_train_onehot = one_hot_encode(Y_train, 100)

        if dataset_name == 'military':
            Y_train_onehot = one_hot_encode(Y_train, 10)
        # Get deep feature representations
        X_train_features = get_deep_representations(model, X_train)
        # X_test_normal_features = get_deep_representations_googlenet(model, X_test)
        # X_test_adv_features = get_deep_representations_googlenet(model, X_test_adv)
        # Train one KDE per class

        class_inds = {}
        for i in range(Y_train_onehot.shape[1]):
            class_inds[i] = np.where(Y_train_onehot.argmax(axis=1) == i)[0]
        kdes = {}
        warnings.warn("Using pre-set kernel bandwidths that were determined "
                      "optimal for the specific CNN models of the paper. If you've "
                      "changed your model, you'll need to re-optimize the "
                      "bandwidth.")
        print('bandwidth %.4f for %s' % (BANDWIDTHS[dataset], dataset))
        for i in range(Y_train_onehot.shape[1]):
            if len(class_inds[i]) == 0:
                print("Warning: No data for class " + str(i) + ". KDE will not be trained for this class")
                continue  # Skip KDE training for this class

            kdes[i] = KernelDensity(kernel='gaussian',
                                    bandwidth=BANDWIDTHS[dataset]) \
                .fit(X_train_features[class_inds[i]])
        dump(kdes, kdes_path)

    return kdes


def write_csv(csv_path, header, datarows):
    # Write data to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write header
        writer.writerows(datarows)  # Write data rows


def train(params):
    Model = params['Model']
    attack_method = params['Attack_Method']

    dataset_name = 'military'

    image_size = (224, 224)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    transform_img = [
        vision.Decode(),
        vision.Resize((256, 256)),
        vision.CenterCrop(image_size),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW(),

    ]
    if Model == 'VGGNet':
        model = vgg16(config.num_classes, config, phase="test")
        load_param_into_net(model, load_checkpoint(
            '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image_mindspore/save_models/0-55_153_military.ckpt'))
        model.add_flags_recursive(fp16=True)

    print("当前测试的攻击方法是: " + attack_method)
    if dataset_name == 'imagenet':
        test_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/imagenet_100/test'
        train_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/imagenet_100/train'
    elif dataset_name == 'military':
        test_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/Military/test'
        train_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/Military/train'

    class_to_label = {}
    # 生成类名到数字标签的映射
    for i, class_folder in enumerate(sorted(glob.glob(os.path.join(test_dataset_dir, "*")))):
        class_name = os.path.basename(class_folder)
        class_to_label[class_name] = i  # 标签映射

    X_test = []
    Y_test = []

    # 遍历 DataLoader
    for class_folder in glob.glob(os.path.join(test_dataset_dir, "*")):
        class_name = class_to_label[os.path.basename(class_folder)]
        class_images = glob.glob(os.path.join(class_folder, "*.JPEG"))
        for class_image in class_images:
            image = np.fromfile(class_image, np.uint8)

            for op in transform_img:
                image = op(image)
            X_test.append(image)
            Y_test.append(class_name)

    adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image_mindspore/dataset/PatchAttack_less_less/adv'
    X_adv = []
    Y_adv = []
    adv_images = []
    for class_folder in glob.glob(os.path.join(adv_dataset_dir, "*")):
        class_name = os.path.basename(class_folder)
        class_images = glob.glob(os.path.join(class_folder, "*.png"))

        for class_image in class_images:
            image = np.fromfile(class_image, np.uint8)
            adv_images.append(class_image)
            for op in transform_img:
                image = op(image)
            X_adv.append(image)
            Y_adv.append(class_name)

    ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image_mindspore/dataset/PatchAttack_less_less/ori'

    X_ori = []
    Y_ori = []
    ori_images = []
    for class_folder in glob.glob(os.path.join(ori_dataset_dir, "*")):
        class_name = os.path.basename(class_folder)
        class_images = glob.glob(os.path.join(class_folder, "*.png"))

        for class_image in class_images:
            image = np.fromfile(class_image, np.uint8)
            ori_images.append(class_image)
            for op in transform_img:
                image = op(image)
            X_ori.append(image)
            Y_ori.append(class_name)

    kdes = get_kd(Model, model, X_test, Y_test, X_ori, X_adv, dataset_name)

    adv_uncertainty_mean = []
    adv_inconsistency_mean = []
    adv_kd = []
    adv_label = []

    clean_uncertainty_mean = []
    clean_inconsistency_mean = []
    clean_kd = []
    clean_label = []

    ori_input_paths = []
    adv_input_paths = []
    csv_path = 'features.csv'

    print('start feature extraction')
    header = ['label', 'input', 'kd']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write header
    for i in range(len(ori_images)):
        ori_image = ori_images[i]
        input = np.fromfile(ori_image, np.uint8)
        for op in transform_img:
            input = op(input)
        input = np.expand_dims(input, axis=0)
        X_test_ori_features = get_deep_representations(model, input)
        output = model(Tensor(input, mstype.float32))
        output = output.asnumpy()

        preds_test_ori = np.argmax(output, (-1))
        densities_ori = score_samples(
            kdes,
            X_test_ori_features,
            preds_test_ori
        )
        del input, X_test_ori_features, output, preds_test_ori  # 释放内存
        gc.collect()  # 强制垃圾回收
        mindspore.ms_memory_recycle()

        # uncertainty_mean=get_bayesian_uncertainty(Model,input)
        # inconsistency_mean = get_bayesian_inconsistency(Model,input)
        #
        # adv_inconsistency_mean.append(inconsistency_mean)
        # adv_uncertainty_mean.append(uncertainty_mean)
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([0, ori_image, densities_ori])  # 追加数据

    for i in range(len(adv_images)):
        adv_image = adv_images[i]
        input = np.fromfile(adv_image, np.uint8)
        for op in transform_img:
            input = op(input)
        input = np.expand_dims(input, axis=0)
        X_test_adv_features = get_deep_representations(model, input)
        output = model(Tensor(input, mstype.float32))
        output = output.asnumpy()

        preds_test_adv = np.argmax(output, (-1))
        densities_adv = score_samples(
            kdes,
            X_test_adv_features,
            preds_test_adv
        )

        # uncertainty_mean = get_bayesian_uncertainty(Model,input)
        # inconsistency_mean = get_bayesian_inconsistency(Model,input)
        #
        # clean_inconsistency_mean.append(inconsistency_mean)
        # clean_uncertainty_mean.append(uncertainty_mean)
        del input, X_test_adv_features, output, preds_test_adv  # 释放内存
        gc.collect()  # 强制垃圾回收
        mindspore.ms_memory_recycle()
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([1, adv_image, densities_adv])  # 追加数据
        # adv_kd.append(densities_adv)
        # adv_label.append(1)
    # print("start writing")
    # print(len(adv_kd), len(adv_images), len(adv_label))
    # print(len(clean_kd), len(ori_images), len(clean_label))
    # header = ['label', 'input', 'kd']
    # datarows = []
    # for kd, input, label in zip(adv_kd, adv_images, adv_label):
    #     datarows.append([label, input, kd])
    # for kd, input, label in zip(clean_kd, ori_images, clean_label):
    #     datarows.append([label, input, kd])
    # write_csv(csv_path, header, datarows)

    jsontext = {
        'Message': '相关特征已写入csv'

    }

    return jsontext


def detect_main(params):
    if params['mode'] == 'train':
        result = train(params)
        return result
    elif params['mode'] == 'test':
        result = train(params)
        return result


if __name__ == "__main__":
    defparam = {
        "taskId": 12345,
        "mode": 'train',
        "Model": 'VGGNet',
        "Attack_Method": "PatchAttack"
    }

    start_time = time.time()
    result = detect_main(defparam)
    end_time = time.time()
    print(end_time-start_time)
    print(result)
