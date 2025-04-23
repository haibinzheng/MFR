import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import torchvision
import os, sys, shutil, time, random, math
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms
import net
import json
import warnings
from torch.utils.data import random_split, DataLoader
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout, Bayes_ensemble
from scalablebdl.prior_reg import PriorRegularizor
from scalablebdl.mean_field import PsiSGD, to_bayesian, to_deterministic
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load
from util_kd import *
from sklearn.neighbors import KernelDensity
import pickle
import time
from loguru import logger
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d

BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00, 'imagenet': 0.15, 'military': 0.45, 'warship': 1.0}
parser = argparse.ArgumentParser(description='Training script for ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', type=str,
                    default='/data/LargeData/Large/ImageNet')

parser.add_argument('--dataset', metavar='DSET', type=str, default='imagenet')
parser.add_argument('--arch', metavar='ARCH', default='resnet50')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--schedule', type=int, nargs='+', default=[3, 6, 9],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1],
                    help='LR for psi is multiplied by gamma on schedule')

# Regularization
parser.add_argument('--decay', type=float, default=1e-4,
                    help='Weight decay')

# Checkpoints
parser.add_argument('--save_path', type=str, default='log_checkpoints',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--job-id', type=str, default='bayesadapter-imagenet')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='Evaluate model on test set')

# Acceleration
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 4)')

# Random seed
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# Bayesian
parser.add_argument('--psi_init_range', type=float, nargs='+', default=[-6, -5])
parser.add_argument('--num_fake', type=int, default=1000)
parser.add_argument('--uncertainty_threshold', type=float, default=0.75)

# Fake generated data augmentation
parser.add_argument('--blur_prob', type=float, default=0.5)
parser.add_argument('--blur_sig', type=float, nargs='+', default=[0., 3.])
parser.add_argument('--jpg_prob', type=float, default=0.5)
parser.add_argument('--jpg_method', type=str, nargs='+', default=['cv2', 'pil'])
parser.add_argument('--jpg_qual', type=int, nargs='+', default=[30, 100])

# Attack settings
parser.add_argument('--epsilon', default=16. / 255., type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=1. / 255., type=float,
                    help='perturb step size')
parser.add_argument('--random', default=True,
                    help='random initialization for PGD')

# Dist
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-port', default='1234', type=str,
                    help='port used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=7, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def one_hot_encode(labels, num_classes):
    # 创建一个形状为 [len(labels), num_classes] 的数组，所有元素初始化为0
    one_hot = np.zeros((len(labels), num_classes))
    # np.arange(len(labels)) 创建一个从0到len(labels)-1的数组
    # labels 是一个包含类别索引的数组
    # 此操作将每行对应的列索引设置为1
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot
def get_image_list(path):
    IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png", ".npy",'.PNG','.JPEG']
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)

    return image_names

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


def evaluteTop1_new(m, x, y):
    m.eval()
    m.to(device)
    batch_size = 16

    correct = 0
    total = 0

    for i in range(0, len(y), batch_size):
        x_batch = x[i:i + batch_size].to(device)
        y_batch = y[i:i + batch_size].to(device)

        with torch.no_grad():
            logits = m(x_batch)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y_batch).sum().item()

        # 清理GPU内存
        x_batch.cpu()
        y_batch.cpu()

        total += y_batch.size(0)

    return correct / total


def evaluteTop5_new(m, x, y):
    m.eval()

    m.to(device)
    batch_size = 16
    correct = 0
    total = 0

    # 分批处理数据
    for i in range(0, len(y), batch_size):
        x_batch = x[i:i + batch_size].to(device)
        y_batch = y[i:i + batch_size].to(device)

        with torch.no_grad():
            logits = m(x_batch)
            maxk = max((1, 5))
            y_resize = y_batch.view(-1, 1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().item()

        # 清理GPU内存
        x_batch.cpu()
        y_batch.cpu()

        total += y_batch.size(0)

    return correct / total


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_missclassified_indices(m, X, Y):
    m.to(device)
    m.eval()  # 确保模型处于评估模式

    missclassified_indices = []

    # 遍历数据集的每个样本
    for i in range(X.size(0)):
        x = X[i].unsqueeze(0).to(device)  # 只处理单个样本，并确保其在GPU上
        y = Y[i].unsqueeze(0).to(device)  # 对应的真实标签

        with torch.no_grad():
            outputs = m(x)
            _, predicted_label = torch.max(outputs, 1)

            # 检查预测是否正确
            if predicted_label.item() != y.item():
                missclassified_indices.append(i)  # 记录误分类的索引

        # 清理GPU内存
        x.cpu()
        y.cpu()

    return missclassified_indices


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
        if dataset_name == 'warship':
            Y_train_onehot = one_hot_encode(Y_train, 2)
        # Get deep feature representations
        X_train_features = get_deep_representations_googlenet(model, X_train)
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


def train(params):
    best_acc = 0
    Model = params['Model']
    attack_method = params['Attack_Method']

    dataset_name = 'military'

    attack_type = 'PoisonFrog'
    batch_size = 16
    args = parser.parse_args()
    if Model == 'AlexNet':
        model = net.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 10)
        model = model.to(device)
        # model.load_state_dict(torch.load('clean_models/AlexNet_imagenet_100.pth'))
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_models/AlexNet_Military.pth'))
    if Model == 'GoogleNet':
        model = net.googlenet(pretrained=False)
        model.fc = nn.Linear(1024, 10)
        model.aux1.auxiliary_linear1 = torch.nn.Linear(768, 10)
        model.aux2.auxiliary_linear1 = torch.nn.Linear(768, 10)
        model = model.to(device)
        # model.load_state_dict(
        #     torch.load('clean_models/GoogleNet_imagenet_100.pth'))
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_models/GoogleNet_Military.pth'))
    if Model == 'VGG16':
        model = net.VGG16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 10)
        model = model.to(device)
        model.load_state_dict(torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_models/VGG16_Military.pth'))
    if Model == 'ResNet50':
        model = net.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 10)
        model = model.to(device)
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_models/ResNet50_Military.pth'))

    if dataset_name == 'imagenet':
        transform_imagenet = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/imagenet100/train'
    if dataset_name == 'military':
        transform_imagenet = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if dataset_name == 'warship':
        transform_imagenet = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    print("当前训练的攻击方法是: " + attack_method)
    # if Model == 'ResNet50':
    #     adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/ResNet50/ALL/adv'
    #     ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/ResNet50/ALL/ori'
    # if Model == 'AlexNet':
    #     # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/AlexNet/All-Attack-ori'
    #     # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/AlexNet/All-Attack-adv'
    #     ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/IndicatorsCoverage/SJS/Sentiment_backdoor/dataset/AlexNet/BadNets/ori'
    #     adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/IndicatorsCoverage/SJS/Sentiment_backdoor/dataset/AlexNet/BadNets/poi'
    # if Model == 'GoogleNet':
    #     # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/GoogleNet/All-Attack-ori'
    #     # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/GoogleNet/All-Attack-adv'
    #     ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/GoogleNet/ALL/ori'
    #     adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/GoogleNet/ALL/adv'
    # if Model == 'VGG16':
    #     ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/VGG16/ALL/ori'
    #     adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/VGG16/ALL/adv'
    #     # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/FGSM-Lambda1/ori'
    #     # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/FGSM-Lambda1/adv'

    dataset = datasets.ImageFolder(
        root='/data/Newdisk/chenjingwen/DT_B4/IndicatorsCoverage/SJS/Sentiment_backdoor/dataset/ResNet50/PoisonFrog',
        transform=transform_imagenet)
    # 设置总数据集大小
    dataset_size = len(dataset)

    # 设置训练集和测试集的比例 (例如：80% 训练集，20% 测试集)
    train_size = int(0.7 * dataset_size)  # 80% 用于训练
    test_size = dataset_size - train_size  # 20% 用于测试

    # 使用 random_split 分割数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cnn_model = binary_model(2).to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.0001, weight_decay=1e-5)

    criterion = torch.nn.CrossEntropyLoss()
    logger.info(f"---training------")
    train_epochs = 40

    Best_Acc = 0
    for epoch in range(train_epochs):
        logger.info(f"---epoch={epoch}---")
        cnn_model.train()
        running_loss = 0
        train_correct = 0
        test_correct = 0
        total = 0
        total111 = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn_model(images)
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
            cnn_model.eval()
            for images111, labels111 in test_loader:
                images111, labels111 = images111.to(device), labels111.to(device)
                outputs111 = cnn_model(images111)
                predicted_test = torch.argmax(outputs111.data, 1)
                test_correct += (predicted_test == labels111).sum().item()
                total111 += images111.shape[0]
        Test_Accuarcy = test_correct / total111
        logger.info(f"epoch={epoch},Test_Accuarcy={test_correct}/{total111}={test_correct / total111}")
        if Test_Accuarcy > Best_Acc and Test_Accuarcy > 0.5:
            Best_Acc = Test_Accuarcy
            logger.info(f"成功保存！！！")
            torch.save(cnn_model.state_dict(),
                       '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/kd_save/save_rf_new/{}_{}_{}_CNN_Classifier.pth'.format(
                           Model, dataset_name, attack_method))
            torch.save(cnn_model.state_dict(),
                       '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/kd_save/save_rf_new/{}_{}_{}_CNN_Classifier_{}.pth'.format(
                           Model, dataset_name, attack_method, Test_Accuarcy))
    jsontext = {
        'Accuracy': accuracy,
        'Model': Model,
        "Attack_Method": attack_method,
        'Save_path': 'kd_save/'
    }

    return jsontext


def detect(params):
    best_acc = 0
    Model = params['Model']
    attack_method = params['Attack_Method']

    dataset_name = 'military'
    detect_result = []
    attack_type = 'PoisonFrog'
    batch_size = 16
    args = parser.parse_args()
    if Model == 'AlexNet':
        model = net.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 10)
        model = model.to(device)
        # model.load_state_dict(torch.load('clean_models/AlexNet_imagenet_100.pth'))
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_models/AlexNet_Military.pth'))
    if Model == 'GoogleNet':
        model = net.googlenet(pretrained=False)
        model.fc = nn.Linear(1024, 10)
        model.aux1.auxiliary_linear1 = torch.nn.Linear(768, 10)
        model.aux2.auxiliary_linear1 = torch.nn.Linear(768, 10)
        model = model.to(device)
        # model.load_state_dict(
        #     torch.load('clean_models/GoogleNet_imagenet_100.pth'))
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_models/GoogleNet_Military.pth'))
    if Model == 'VGG16':
        model = net.VGG16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 10)
        model = model.to(device)
        model.load_state_dict(torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_models/VGG16_Military.pth'))
    if Model == 'ResNet50':
        model = net.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 2)
        model = model.to(device)
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_models/ResNet50_warship.pth'))

    if dataset_name == 'imagenet':
        transform_imagenet = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/imagenet100/train'
    if dataset_name == 'military':
        transform_imagenet = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if dataset_name == 'warship':
        transform_imagenet = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    print("当前测试的攻击方法是: " + attack_method)
    if Model == 'ResNet50':
        adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/selected_2/warship/ResNet50/ALL_V2/adv'
        ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/selected_2/warship/ResNet50/ALL_V2/ori'
    if Model == 'AlexNet':
        # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/AlexNet/All-Attack-ori'
        # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/AlexNet/All-Attack-adv'
        ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/AlexNet/ALL/ori'
        adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/AlexNet/ALL/adv'
    if Model == 'GoogleNet':
        # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/GoogleNet/All-Attack-ori'
        # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/GoogleNet/All-Attack-adv'
        ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/GoogleNet/ALL/ori'
        adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/GoogleNet/ALL/adv'
    if Model == 'VGG16':
        ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/All-Attack-ori-2'
        adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/All-Attack-adv-2'
        # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/FGSM-Lambda1/ori'
        # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/FGSM-Lambda1/adv'

    test_dataset = datasets.ImageFolder(
        root='/data/Newdisk/chenjingwen/DT_B4/IndicatorsCoverage/SJS/Sentiment_backdoor/dataset/ResNet50/PoisonFrog_less',
        transform=transform_imagenet)


    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cnn_model = binary_model(2).to(device)
    cnn_model.load_state_dict(torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/kd_save/save_rf_new/ResNet50_military_PoisonFrog_CNN_Classifier.pth'))
    cnn_model.eval()

    test_correct = 0
    total111 = 0

   
    path333 = '/data/Newdisk/chenjingwen/DT_B4/IndicatorsCoverage/SJS/Sentiment_backdoor/dataset/ResNet50/PoisonFrog_less'
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
        outputs111 = cnn_model(data)

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
            #
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
            
            detect_result.append(item)
        test_correct += (predicted_test == target).sum().item()
        total111 += data.shape[0]
        
    end_time = time.time()
    everage_time = (end_time - start_time) / total111
    Test_Accuarcy = test_correct / total111
    logger.info(f"CNN_Classifier_Detect:Test_Accuarcy={test_correct}/{total111}={test_correct / total111}")
    logger.info(f"everage_time={everage_time},All_Attack_Methods={attack_method}")


    jsontext = {
        'Accuracy': Test_Accuarcy,
        "Attack_Method": attack_method,
        'average_time': everage_time,
        "detect_result": detect_result
    }




    return jsontext

def detect_main(params):
    if params['mode'] == 'train':
        result = train(params)
        return result
    elif params['mode'] == 'test':
        result = detect(params)
        return result


if __name__ == "__main__":
    defparam = {
        "taskId": 12345,
        "mode": 'test',
        "Model": 'ResNet50',
        "Attack_Method": "PoisonFrog"
    }
    print(defparam)

    result = detect_main(defparam)
    print(result)
