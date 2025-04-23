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
    # if attack_method in ['FGSM-Lambda1', 'DeepFool-Lambda1', 'PatchAttack', 'NFA-Lambda1', 'MIFGSM-Lambda1',
    #                      'JSMA-Lambda1', 'FGSM-L1', 'EAD-Lambda1', 'Adef-Lambda1', 'PGD-Lambda1',
    #                      'CarliniWagnerL2Attack']:
    #     attack_type = 'WhiteBox'
    #
    # else:
    #     attack_type = 'BlackBox'
    attack_type = 'ALL-v2'
    batch_size = 1
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
    if Model == 'ResNet50':
        adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/ResNet50/ALL-V2/adv'
        ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/ResNet50/ALL-V2/ori'
    if Model == 'AlexNet':
        # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/AlexNet/All-Attack-ori'
        # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/AlexNet/All-Attack-adv'
        ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/AlexNet/FGSM/ori'
        adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/AlexNet/FGSM/adv'
    if Model == 'GoogleNet':
        # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/GoogleNet/All-Attack-ori'
        # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/GoogleNet/All-Attack-adv'
        ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/GoogleNet/ALL/ori'
        adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/Military/GoogleNet/ALL/adv'
    if Model == 'VGG16':
        ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/VGG16/ALL/ori'
        adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/VGG16/ALL/adv'
        # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/FGSM-Lambda1/ori'
        # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/VGG16/FGSM-Lambda1/adv'

    advset = torchvision.datasets.ImageFolder(
        root=os.path.join(adv_dataset_dir, ''),
        transform=transform_imagenet)
    adv_loader = torch.utils.data.DataLoader(advset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    # 初始化列表来存储数据和标签
    X_adv = []
    Y_adv = []

    # 遍历 DataLoader
    for images, labels in adv_loader:
        X_adv.append(images)
        Y_adv.append(labels)

    # 将列表中的数据合并成单个张量
    X_adv = torch.cat(X_adv, dim=0)  # 沿着第一个维度（批次大小）合并
    Y_adv = torch.cat(Y_adv, dim=0)

    # ori_dataset_dir = 'attack_datasets/{}_data/{}All/ori'.format(Model, attack_type)

    oriset = torchvision.datasets.ImageFolder(
        root=os.path.join(ori_dataset_dir, ''),
        transform=transform_imagenet)
    ori_loader = torch.utils.data.DataLoader(oriset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    # 初始化列表来存储数据和标签
    X_ori = []
    Y_ori = []

    for images, labels in ori_loader:
        X_ori.append(images)
        Y_ori.append(labels)

        # 将列表中的数据合并成单个张量
    X_ori = torch.cat(X_ori, dim=0)  # 沿着第一个维度（批次大小）合并
    Y_ori = torch.cat(Y_ori, dim=0)

    if dataset_name == 'imagenet':
        test_dataset_dir = 'clean_datasets/imagenet_100/test'
        train_dataset_dir = 'clean_datasets/imagenet_100/train'
    elif dataset_name == 'military':
        test_dataset_dir = 'clean_datasets/Military/test'
        train_dataset_dir = 'clean_datasets/Military/train'
    elif dataset_name == 'warship':
        test_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/warship/test'
        train_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/warship/train'

    testset = torchvision.datasets.ImageFolder(
        root=os.path.join(test_dataset_dir, ''),
        transform=transform_imagenet)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    X_test = []
    Y_test = []

    # 遍历 DataLoader
    for images, labels in test_loader:
        X_test.append(images)
        Y_test.append(labels)

    # 将列表中的数据合并成单个张量
    X_test = torch.cat(X_test, dim=0)  # 沿着第一个维度（批次大小）合并
    Y_test = torch.cat(Y_test, dim=0)

    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(train_dataset_dir, ''),
        transform=transform_imagenet)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=0)

    # BNN transform
    bnn_modelpath = 'kd_save/save_bnn/{}_bnn_{}.pth'.format(Model, dataset_name)
    # bnn_modelpath = 'save_bnn/finetune_VGG_whitebox.pth'
    if os.path.isfile(bnn_modelpath):
        print("BNN模型已经存在")
        bayesian_net = to_bayesian(model)
        bayesian_net.load_state_dict(torch.load(bnn_modelpath))

    else:
        print("模型路径不存在，重新训练一个")
        disable_dropout(model)
        bayesian_net = to_bayesian(model)
        unfreeze(bayesian_net)
        torch.save(bayesian_net.state_dict(), bnn_modelpath)

    criterion = torch.nn.CrossEntropyLoss().cuda(7)
    bayesian_net.cuda(7)
    bayesian_net.eval()
    # 这边测的是原模型而不是贝叶斯模型。贝叶斯模型的两次预测之间有较大的不一致
    # missclassified_indices = get_missclassified_indices(model, X_adv, Y_adv)
    # X_ori = X_ori[missclassified_indices]
    # Y_ori = Y_ori[missclassified_indices]
    #
    # X_adv = X_adv[missclassified_indices]
    # Y_adv = Y_adv[missclassified_indices]

    print("...test on original dataset...")

    # ori_top1 = evaluteTop1_new(model, X_ori, Y_ori)
    # print("攻击前TOP1: " + str(ori_top1))
    #
    # print("...test on adversarial dataset...")
    #
    # adv_top1 = evaluteTop1_new(model, X_adv, Y_adv)
    #
    # print("攻击后TOP1: " + str(adv_top1))

    # 创建 Dataset 对象
    dataset_ori = CustomDataset(X_ori, Y_ori)
    dataset_adv = CustomDataset(X_adv, Y_adv)

    ori_loader_new = torch.utils.data.DataLoader(dataset_ori, batch_size=batch_size, shuffle=True)
    adv_loader_new = torch.utils.data.DataLoader(dataset_adv, batch_size=batch_size, shuffle=True)

    kdes = get_kd(Model, model, X_test, Y_test, X_ori, X_adv, dataset_name)

    result = []
    adv_uncertainty_mean = []
    adv_inconsistency_mean = []
    adv_kd = []
    adv_label = []
    clean_uncertainty_mean = []
    clean_inconsistency_mean = []
    clean_kd = []
    clean_label = []

    ood_train_loader_iter = iter(adv_loader_new)

    for i, (input, target) in enumerate(ori_loader_new):
        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        try:
            input1 = next(ood_train_loader_iter)[0]
        except StopIteration:  # 报错了，加上这个异常处理
            ood_train_loader_iter = iter(adv_loader_new)
            input1 = next(ood_train_loader_iter)[0]

        input1 = input1.cuda(args.gpu, non_blocking=True)
        # KD计算

        model.cuda(7)
        preds_test_normal = get_predict_class(model, input)
        preds_test_adv = get_predict_class(model, input1)

        X_test_normal_features = get_deep_representations_googlenet(model, input)
        X_test_adv_features = get_deep_representations_googlenet(model, input1)

        densities_normal = score_samples(
            kdes,
            X_test_normal_features,
            preds_test_normal
        )

        densities_adv = score_samples(
            kdes,
            X_test_adv_features,
            preds_test_adv
        )

        adv_kd.append(densities_adv)
        clean_kd.append(densities_normal)

        # KD结束

        bs = input.shape[0]
        bs1 = input1.shape[0]

        # repeat(2, 1, 1, 1)用于后续衡量模型对同一输入数据进行两次处理后输出结果的一致性
        output = bayesian_net(torch.cat([input.repeat(2, 1, 1, 1), input1.repeat(2, 1, 1, 1)]))

        out0_0 = output[:bs].softmax(-1)
        out0_1 = output[bs:bs + bs].softmax(-1)
        out1_0 = output[bs + bs:bs + bs + bs1].softmax(-1)
        out1_1 = output[bs + bs + bs1:].softmax(-1)

        mi0 = ent((out0_0 + out0_1) / 2.) - (ent(out0_0) + ent(out0_1)) / 2.
        mi1 = ent((out1_0 + out1_1) / 2.) - (ent(out1_0) + ent(out1_1)) / 2.
        # print("out1_0的不确定性：" + str(ent(out1_0).detach().item()))
        # print("out1_1的不确定性：" + str(ent(out1_1).detach().item()))
        # print("对抗数据两次预测不确定性平均值：" + str(((ent(out1_0) + ent(out1_1)) / 2.).detach().item()))
        # print("对抗数据两次预测的结果一致性高低：" + str(mi1.detach().item()))
        adv_uncertainty_mean.append(((ent(out1_0) + ent(out1_1)) / 2.).detach().item())
        adv_inconsistency_mean.append(mi1.detach().item())
        adv_label.append(1)
        # print("out0_0的不确定性：" + str(ent(out0_0).detach().item()))
        # print("out0_1的不确定性：" + str(ent(out0_1).detach().item()))
        # print("干净数据两次预测不确定性平均值：" + str(((ent(out0_0) + ent(out0_1)) / 2.).detach().item()))
        # print("干净两次预测的结果一致性高低：" + str(mi0.detach().item()))
        clean_uncertainty_mean.append(((ent(out0_0) + ent(out0_1)) / 2.).detach().item())
        clean_inconsistency_mean.append(mi0.detach().item())
        clean_label.append(0)
        # print("..................")

        # 将数据转换为 numpy 数组
        # 合并特征和标签数组
    X = np.array(adv_uncertainty_mean + clean_uncertainty_mean).reshape(-1, 1)  # 将数据转换为二维数组，符合sklearn的要求
    y = np.array(adv_label + clean_label)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_1 = np.array(adv_inconsistency_mean + clean_inconsistency_mean).reshape(-1, 1)

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y, test_size=0.2, random_state=42)

    X_2 = np.array(adv_kd + clean_kd).reshape(-1, 1)
    scaler_path = 'kd_save/save_scaler_new/scaler_{}_{}_{}.pkl'.format(Model, attack_type, dataset_name)
    if os.path.isfile(scaler_path):
        print("scaler已经存在")
        fp = open(scaler_path, "rb+")
        scaler = pickle.load(fp)
        X_2 = scaler.transform(X_2)
    else:
        print("scaler路径不存在，重新训练一个")
        scaler = MinMaxScaler().fit(X_2)
        X_2 = scaler.transform(X_2)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y, test_size=0.2, random_state=42)

    X_combined = np.hstack((X, X_1, X_2))

    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y,
                                                                                            test_size=0.2,
                                                                                            random_state=42)

    # print("使用三者结合进行预测：")
    model_filename = 'kd_save/save_rf_new/rf_{}_{}_{}.pkl'.format(Model, attack_type, dataset_name)
    # model_filename = 'save_rf/rf_{}_{}_finetune.pkl'.format(Model, attack_type)
    if os.path.isfile(model_filename):
        print("分类模型已经存在")
        classifier_2 = load(model_filename)
    else:
        print("分类模型路径不存在，重新训练一个")
        classifier_2 = RandomForestClassifier(n_estimators=100)
        classifier_2.fit(X_train_combined, y_train_combined)
        dump(classifier_2, model_filename)

    y_pred = classifier_2.predict(X_test_combined)
    accuracy = accuracy_score(y_test_combined, y_pred)

    result.append(accuracy)
    cm = confusion_matrix(y_test_combined, y_pred)

    print("confusion Matrix: ")
    print(cm)
    print("Accuracy with Random Forest:", accuracy)

    print(attack_method + '的最终ACC是：' + str(sum(result) / len(result)))

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

    dataset_name = 'warship'

    # if attack_method in ['FGSM-Lambda1', 'DeepFool-Lambda1', 'PatchAttack', 'NFA-Lambda1', 'MIFGSM-Lambda1',
    #                      'JSMA-Lambda1', 'FGSM-L1', 'EAD-Lambda1', 'Adef-Lambda1', 'PGD-Lambda1',
    #                      'CarliniWagnerL2Attack']:
    #     attack_type = 'WhiteBox'
    #
    # else:
    #     attack_type = 'BlackBox'

    attack_type = 'ALL-v2'

    device = torch.device("cuda:7")
    batch_size = 1
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
        model.fc = nn.Linear(1024, 2)
        model.aux1.auxiliary_linear1 = torch.nn.Linear(768, 2)
        model.aux2.auxiliary_linear1 = torch.nn.Linear(768, 2)
        model = model.to(device)
        # model.load_state_dict(
        #     torch.load('clean_models/GoogleNet_imagenet_100.pth'))
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/model_data/googlenet_ship.pth'))
    if Model == 'VGG16':
        model = net.VGG16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 2)
        model = model.to(device)
        model.load_state_dict(torch.load('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/model_data/vgg16_ship.pth'))
    if Model == 'ResNet50':
        model = net.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 2)
        model = model.to(device)
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_models/ResNet50_warship.pth'))

    if dataset_name == 'imagenet':
        transform_imagenet = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
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

        # ori_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/AlexNet/FGSM-Lambda1/ori'
        # adv_dataset_dir = '/data/Newdisk/chenjingwen/qkh/military_adv_gen/adv_examples_images/warship/AlexNet/FGSM-Lambda1/adv'
    ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/selected_2/warship/ResNet50/FGSM-Lambda1/png/ori'
    adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/selected_2/warship/ResNet50/FGSM-Lambda1/png/adv'
    # output_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/selected_2/{}/{}'.format(dataset_name, Model)
    print("当前测试的攻击方法是: " + attack_method)

    advset = torchvision.datasets.ImageFolder(
        root=os.path.join(adv_dataset_dir, ''),
        transform=transform_imagenet)
    adv_loader = torch.utils.data.DataLoader(advset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    # ori_dataset_dir = 'selected_attack_datasets/{}/{}/ori_png'.format(
    #     Model,
    #     attack_method)
    # ori_dataset_dir = '/data0/BigPlatform/ZJPlatform/010-ALLData/001_GenData/000_Image/000_Dataset/0/Image/Attack/Military/VGG16/{}/png/ori'.format(attack_method)

    oriset = torchvision.datasets.ImageFolder(
        root=os.path.join(ori_dataset_dir, ''),
        transform=transform_imagenet)
    ori_loader = torch.utils.data.DataLoader(oriset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=0)

    # BNN transform
    bnn_modelpath = 'kd_save/save_bnn/{}_bnn_{}.pth'.format(Model,dataset_name)

    bayesian_net = to_bayesian(model)
    bayesian_net.load_state_dict(torch.load(bnn_modelpath))

    criterion = torch.nn.CrossEntropyLoss().cuda(7)
    bayesian_net.cuda(7)
    bayesian_net.eval()

    kdes = get_kd(Model, model, [], [], [], [], dataset_name)

    result = []
    input_clean = []
    input_adv = []
    input_clean_label = []
    input_adv_label = []

    adv_uncertainty_mean = []
    adv_inconsistency_mean = []
    adv_kd = []
    adv_label = []
    clean_uncertainty_mean = []
    clean_inconsistency_mean = []
    clean_kd = []
    clean_label = []

    ood_train_loader_iter = iter(adv_loader)
    start_time = time.time()
    for i, (input, target) in enumerate(ori_loader):
        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        try:
            next_iter = next(ood_train_loader_iter)
            input1 = next_iter[0]
            target1 = next_iter[1]
        except StopIteration:  # 报错了，加上这个异常处理
            ood_train_loader_iter = iter(adv_loader)
            next_iter = next(ood_train_loader_iter)
            input1 = next_iter[0]
            target1 = next_iter[1]

        input1 = input1.cuda(args.gpu, non_blocking=True)
        # KD计算

        model.cuda(7)
        preds_test_normal = get_predict_class(model, input)
        preds_test_adv = get_predict_class(model, input1)

        X_test_normal_features = get_deep_representations_googlenet(model, input)
        X_test_adv_features = get_deep_representations_googlenet(model, input1)
        input_clean.append(input)
        input_adv.append(input1)
        input_clean_label.append(target.cpu().numpy()[0])
        input_adv_label.append(target1.cpu().numpy()[0])
        print("kde................")
        print(kdes)
        print(X_test_normal_features)
        print(len(X_test_normal_features))
        print(len(X_test_normal_features[0]))
        print(preds_test_normal)
        densities_normal = score_samples(
            kdes,
            X_test_normal_features,
            preds_test_normal
        )

        densities_adv = score_samples(
            kdes,
            X_test_adv_features,
            preds_test_adv
        )

        adv_kd.append(densities_adv)
        clean_kd.append(densities_normal)

        # KD结束

        bs = input.shape[0]
        bs1 = input1.shape[0]

        # repeat(2, 1, 1, 1)用于后续衡量模型对同一输入数据进行两次处理后输出结果的一致性
        output = bayesian_net(torch.cat([input.repeat(2, 1, 1, 1), input1.repeat(2, 1, 1, 1)]))

        out0_0 = output[:bs].softmax(-1)
        out0_1 = output[bs:bs + bs].softmax(-1)
        out1_0 = output[bs + bs:bs + bs + bs1].softmax(-1)
        out1_1 = output[bs + bs + bs1:].softmax(-1)

        mi0 = ent((out0_0 + out0_1) / 2.) - (ent(out0_0) + ent(out0_1)) / 2.
        mi1 = ent((out1_0 + out1_1) / 2.) - (ent(out1_0) + ent(out1_1)) / 2.
        # print("out1_0的不确定性：" + str(ent(out1_0).detach().item()))
        # print("out1_1的不确定性：" + str(ent(out1_1).detach().item()))
        # print("对抗数据两次预测不确定性平均值：" + str(((ent(out1_0) + ent(out1_1)) / 2.).detach().item()))
        # print("对抗数据两次预测的结果一致性高低：" + str(mi1.detach().item()))
        adv_uncertainty_mean.append(((ent(out1_0) + ent(out1_1)) / 2.).detach().item())
        adv_inconsistency_mean.append(mi1.detach().item())
        adv_label.append(1)
        # print("out0_0的不确定性：" + str(ent(out0_0).detach().item()))
        # print("out0_1的不确定性：" + str(ent(out0_1).detach().item()))
        # print("干净数据两次预测不确定性平均值：" + str(((ent(out0_0) + ent(out0_1)) / 2.).detach().item()))
        # print("干净两次预测的结果一致性高低：" + str(mi0.detach().item()))
        clean_uncertainty_mean.append(((ent(out0_0) + ent(out0_1)) / 2.).detach().item())
        clean_inconsistency_mean.append(mi0.detach().item())
        clean_label.append(0)
        # print("..................")

        # 将数据转换为 numpy 数组
        # 合并特征和标签数组
    input_X = np.array(input_clean + input_adv)
    target_X = np.array(input_clean_label + input_adv_label)
    X = np.array(adv_uncertainty_mean + clean_uncertainty_mean).reshape(-1, 1)  # 将数据转换为二维数组，符合sklearn的要求
    y = np.array(adv_label + clean_label)

    X_1 = np.array(adv_inconsistency_mean + clean_inconsistency_mean).reshape(-1, 1)

    X_2 = np.array(adv_kd + clean_kd).reshape(-1, 1)
    scaler_path = 'kd_save/save_scaler/scaler_{}_{}_{}.pkl'.format(Model, attack_type,dataset_name)

    fp = open(scaler_path, "rb+")
    scaler = pickle.load(fp)
    X_2 = scaler.transform(X_2)

    X_combined = np.hstack((X, X_1, X_2))

    # print("使用三者结合进行预测：")
    model_filename = 'kd_save/save_rf/rf_{}_{}_{}.pkl'.format(Model, attack_type,dataset_name)

    classifier_2 = load(model_filename)
    print(X_combined)
    y_pred = classifier_2.predict(X_combined)
    end_time = time.time()
    accuracy = accuracy_score(y, y_pred)

    result.append(accuracy)
    cm = confusion_matrix(y, y_pred)
    everage_time = (end_time - start_time) / len(y)

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    # for i in range(len(y)):
    #     data = input_X[i]
    #     if y[i] == y_pred[i]:
    #
    #         if y[i] == 1:
    #             save_image = inv_normalize(data.squeeze(0))
    #             # print(input1)
    #             save_image = save_image.permute(1, 2, 0)  # 将(C, H, W)转为(H, W, C)
    #             save_image = torch.clamp(save_image, 0, 1)  # 将值限制在[0, 1]之间
    #             array = save_image.cpu().numpy()
    #             array = (array * 255).astype(np.uint8)
    #             img = Image.fromarray(array)
    #             os.makedirs(output_dir + f'/ori/{target_X[i]}', exist_ok=True)
    #             img.save(os.path.join(output_dir + f'/ori/{target_X[i]}/img_{i}.png'))
    #         if y[i] == 0:
    #             save_image = inv_normalize(data.squeeze(0))
    #             # print(input1)
    #             save_image = save_image.permute(1, 2, 0)  # 将(C, H, W)转为(H, W, C)
    #             save_image = torch.clamp(save_image, 0, 1)  # 将值限制在[0, 1]之间
    #             array = save_image.cpu().numpy()
    #             array = (array * 255).astype(np.uint8)
    #             img = Image.fromarray(array)
    #             os.makedirs(output_dir + f'/adv/{target_X[i]}', exist_ok=True)
    #             img.save(os.path.join(output_dir + f'/adv/{target_X[i]}/img_{i}.png'))

    print(everage_time)
    print("onfusion Matrix: ")
    print(cm)
    print("Accuracy with Random Forest:", accuracy)
    jsontext = {
        'Accuracy': accuracy,
        'Model': Model,
        "Attack_Method": attack_method,
        'average_time': everage_time,
        "detect_result": y_pred.tolist()

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
        "mode": 'train',
        "Model": 'ResNet50',
        "Attack_Method": "ALL-v2"
    }
    print(defparam)

    result = detect_main(defparam)
    print(result)
