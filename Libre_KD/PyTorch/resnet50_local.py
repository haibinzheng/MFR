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


os.environ["CUDA_VISABLE_DEVICES"] = "1,2,3,4,5,6,7"
torch.cuda.set_device(4)
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")




def model_load(model_path, type):
    if type == 'bnn':
        model = net.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 10)
        model = to_bayesian(model)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
    if type == 'kde':
        model = load(model_path)
    if type == 'classifier':
        model = load(model_path)
        
    if type == 'scaler':
        fp = open(model_path, "rb+")
        model = pickle.load(fp)
    if type == 'ori':
        model = net.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 10)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

    return model


def resnet_fgsm_detection(image, model):
    img = Image.open(image)
    bnn_model = model[0]
    kde = model[1]
    classifier = model[2]
    scaler = model[3]
    ori_model = model[4]

    feature = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transformed_image = transform(img).unsqueeze(0).to(device)
    X_test_features = get_deep_representations_googlenet(ori_model, transformed_image)
    preds_test_normal = get_predict_class(ori_model, transformed_image)
    densities = score_samples(
        kde,
        X_test_features,
        preds_test_normal
    )
    feature.append(densities)

    bs = transformed_image.shape[0]
    output = bnn_model(transformed_image.repeat(2, 1, 1, 1))
    out0 = output[:bs].softmax(-1)
    out1 = output[bs:].softmax(-1)

    
    mi = ent((out0 + out1) / 2.) - (ent(out0) + ent(out1)) / 2.

    feature.append(((ent(out1) + ent(out0)) / 2.).detach().item())
    feature.append(mi.detach().item())

    feature = [feature[1], feature[2], scaler.transform([feature[0]])[0, 0]]
    print(feature)
    pred = classifier.predict(np.array(feature).reshape(1, -1))[0]

    return pred



if __name__ == "__main__":
    FGSM_IMAGES_PATH = "/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/gen_data/Military/ResNet50/ZOO-Lambda1/png"  # 测试数据集路径

    RESNET50_MODEL_PATH = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/model_data/ResNet50_Military.pth'
    RESNET_BNN_MODEL_PATH = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/kd_save/save_bnn/ResNet_bnn.pth'
    RESNET_KDE_PATH = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/kd_save/save_kdes/kdes_military_ResNet.joblib'
    RESNET_CLASSIFIER_PATH = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/kd_save/save_rf/rf_ResNet_ALL.pkl'
    RESNET_SCALER_PATH = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/kd_save/save_scaler/scaler_ResNet_ALL.pkl'
    model = []

    RESNET_BNN_MODEL = model_load(RESNET_BNN_MODEL_PATH, 'bnn')
    RESNET_KDE = model_load(RESNET_KDE_PATH, 'kde')
    RESNET_CLASSIFIER = model_load(RESNET_CLASSIFIER_PATH, 'classifier')
    RESNET_SCALER = model_load(RESNET_SCALER_PATH, 'scaler')
    RESNET50_MODEL = model_load(RESNET50_MODEL_PATH, 'ori')
    model.append(RESNET_BNN_MODEL)
    model.append(RESNET_KDE)
    model.append(RESNET_CLASSIFIER)
    model.append(RESNET_SCALER)
    model.append(RESNET50_MODEL)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    results = []
    labels = []
    total_time = 0

    for root, dirs, files in os.walk(FGSM_IMAGES_PATH):  # 以该数据集为例，可替换

        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)

                if 'ori' in file_path.split('/'):
                    cur = 0
                    labels.append(0)
                else:
                    cur = 1
                    labels.append(1)

                with open(file_path, "rb") as img_file:
                    start_time = time.time()
                    result = resnet_fgsm_detection(img_file, model)
                    end_time = time.time()
                    results.append(result)
                    total_time += (end_time - start_time)

                    # if result == cur:
                    #     path_parts = root.split('/')
                    #     # 取出需要的部分（索引为6到最后的部分）
                    #     desired_path = '/'.join(path_parts[9:])
                    #     save_path = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/selected_2/' + desired_path
                    #     print(save_path)
                    #     if not os.path.exists(save_path):
                    #         os.makedirs(save_path)
                    #     shutil.copy(file_path, save_path)

                    

    correct_predictions = sum(p == l for p, l in zip(results, labels))
    accuracy = correct_predictions / len(labels)
    avg_time = total_time / len(labels)
    print("ACC: " + str(accuracy))
    print("avg_time: " + str(avg_time))
