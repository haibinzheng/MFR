import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import net
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout, Bayes_ensemble
from scalablebdl.prior_reg import PriorRegularizor
from scalablebdl.mean_field import PsiSGD, to_bayesian, to_deterministic
import pandas as pd
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load
import shutil
import time
device = torch.device("cuda:2")
im_width, im_height = 224, 224


def ent(prob):
    # 计算熵，添加数值稳定性处理
    epsilon = 1e-15
    prob = torch.clamp(prob, epsilon, 1.0 - epsilon)
    return -(prob * prob.log()).sum(-1)


def normalize_image(image_tensor):
    # 定义均值和标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 确保输入张量的形状为 (1, 3, H, W)
    if image_tensor.shape[0] != 1 or image_tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (1, 3, H, W)")

    # 归一化处理
    normalized_tensor = (image_tensor.cpu() - mean[:, None, None]) / std[:, None, None]

    return normalized_tensor.to(device)


def get_torch_models(Model):
    if Model == 'AlexNet':
        model = net.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 2)
        model = model.to(device)
        model.load_state_dict(
            torch.load('/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/model_data/alexnet_ship.pth'))
        model.eval()
    if Model == 'GoogleNet':
        model = net.googlenet(pretrained=False)
        model.fc = nn.Linear(1024, 100)
        model = model.to(device)
        model.load_state_dict(
            torch.load('kd_save/clean_models/GoogleNet_imagenet_100.pth'))
    if Model == 'VGGNet':
        model = net.VGG16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 10)
        model = model.to(device)
        model.load_state_dict(torch.load('kd_save/clean_models/VGG16_Military.pth'))
    if Model == 'ResNet':
        model = net.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 10)
        model = model.to(device)
        model.load_state_dict(torch.load('kd_save/clean_models/ResNet50_Military.pth'))

    return model


def get_bayesian_uncertainty(Model, input):
    model = get_torch_models(Model)
    bnn_modelpath = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/kd_save/save_bnn/{}_bnn_warship.pth'.format(Model)
    bayesian_net = to_bayesian(model)
    bayesian_net.load_state_dict(torch.load(bnn_modelpath))
    bayesian_net.eval()
    bs = input.shape[0]
    output = bayesian_net(input.repeat(2, 1, 1, 1))

    out0 = output[:bs].softmax(-1)
    out1 = output[bs:].softmax(-1)

    mi = ent((out0 + out1) / 2.) - (ent(out0) + ent(out1)) / 2.
    uncertainty = ((ent(out0) + ent(out1)) / 2.).detach().item()

    return uncertainty


def get_bayesian_inconsistency(Model, input):
    model = get_torch_models(Model)
    bnn_modelpath = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/kd_save/save_bnn/{}_bnn_warship.pth'.format(Model)
    bayesian_net = to_bayesian(model)
    bayesian_net.load_state_dict(torch.load(bnn_modelpath))
    bayesian_net.eval()
    bs = input.shape[0]
    output = bayesian_net(input.repeat(2, 1, 1, 1))

    out0 = output[:bs].softmax(-1)
    out1 = output[bs:].softmax(-1)

    mi = (ent((out0 + out1) / 2.) - (ent(out0) + ent(out1)) / 2.).item()
    uncertainty = ((ent(out0) + ent(out1)) / 2.).detach().item()

    return mi


def train(params):
    Model = params['Model']
    attack_method = params['Attack_Method']

    dataset_name = 'military'

    csv_path = 'features_{}.csv'.format(dataset_name)
    # Read CSV into DataFrame
    # df = pd.read_csv(csv_path)
    #
    # # Extract columns from DataFrame
    # labels = df['label'].tolist()
    # inputs_path = df['input'].tolist()
    # kds = df['kd'].tolist()
    # uncertaintys = []
    # inconsistencys = []
    #
    # for i in range(len(labels)):
    #     label = labels[i]
    #     input_path = inputs_path[i]
    #
    #     img = Image.open(input_path)
    #     if img.mode != 'RGB':
    #         img = img.convert('RGB')
    #     img = img.resize((im_width, im_height))
    #     img = np.array(img) / 255.
    #     img = np.expand_dims(img, 0)
    #     img = torch.tensor(img).to(device)
    #     img = img.permute(0, 3, 1, 2)
    #     img = normalize_image(img)
    #
    #     uncertainty = get_bayesian_uncertainty(Model, img)
    #     uninsistency = get_bayesian_inconsistency(Model, img)
    #     uncertaintys.append(uncertainty)
    #     inconsistencys.append(uninsistency)
    #
    # new_data = {'uncertainty': uncertaintys, 'inconsistency': inconsistencys}
    #
    # # 将字典转换为 DataFrame
    # new_df = pd.DataFrame(new_data)
    #
    # # 将新的 DataFrame 与原始 DataFrame 合并
    # df = pd.concat([df, new_df], axis=1)
    #
    # # 保存 DataFrame 到 CSV 文件
    # df.to_csv('features.csv', index=False)
    acc = get_train_test_data(params, csv_path)
    jsontext = {
        'Accuracy': acc,
        'Model': Model,
        "Attack_Method": attack_method,

    }

    return jsontext


def get_train_test_data(params, csv_path):
    Model = params['Model']
    attack_method = params['Attack_Method']

    dataset_name = 'military'

    df = pd.read_csv(csv_path)

    # Extract columns from DataFrame
    labels = df['label'].tolist()
    inputs_path = df['input'].tolist()
    kds = df['kd'].tolist()
    # uncertaintys = df['uncertainty'].tolist()
    # inconsistencys = df['inconsistency'].tolist()

    Y = []
    X = []
    X_2 = []
    kds_list = []
    for i in range(len(kds)):
        Y.append(int(labels[i]))
        feature = []
        # feature.append(float(uncertaintys[i]))
        # feature.append(float(inconsistencys[i]))
        kds_list.append([float(kds[i].strip('[]'))])
        # X.append(feature)

    scaler_path = 'kd_save/save_scaler/scaler_{}_{}_{}.pkl'.format(Model, attack_method, dataset_name)
    if os.path.isfile(scaler_path):
        print("scaler已经存在")
        fp = open(scaler_path, "rb+")
        scaler = pickle.load(fp)
        kds_2 = scaler.transform(kds_list)
    else:
        print("scaler路径不存在，重新训练一个")
        scaler = MinMaxScaler().fit(kds_list)
        kds_2 = scaler.transform(kds_list)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    # for i in range(len(kds_2)):
    #     cur_feature = X[i]
    #     cur_feature.append(kds_2[i])
    #     X_2.append(cur_feature)

    # X_train, X_test, y_train, y_test = train_test_split(kds_2, Y,
    #                                                     test_size=0.8,
    #                                                     random_state=42)
    X_train, X_test, y_train, y_test = kds_2, kds_2, Y, Y  # 这里所有样本都作为测试集

    # print("使用三者结合进行预测：")
    model_filename = 'kd_save/save_rf/rf_{}_{}_{}.pkl'.format(Model, attack_method, dataset_name)
    # model_filename = 'save_rf/rf_{}_{}_finetune.pkl'.format(Model, attack_type)
    if os.path.isfile(model_filename):
        print("分类模型已经存在")
        classifier_2 = load(model_filename)
    else:
        print("分类模型路径不存在，重新训练一个")
        classifier_2 = RandomForestClassifier(n_estimators=100)
        classifier_2.fit(X_train, y_train)
        dump(classifier_2, model_filename)

    y_pred = classifier_2.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print("confusion Matrix: ")
    print(cm)
    print("Accuracy with Random Forest:", accuracy)

    # adv_saved=0
    # ori_saved=0
    # correct_samples_folder = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image_tf/dataset/military/AlexNet/FGSM_less_less'  # 你可以指定新的文件夹路径
    # if not os.path.exists(correct_samples_folder):
    #     os.makedirs(correct_samples_folder)
    #
    # # 保存判断正确的样本到目标文件夹
    # correct_samples = []
    # for i in range(len(y_pred)):
    #     if y_pred[i] == y_test[i]:  # 判断是否正确
    #         # 获取输入路径和文件名
    #         input_path = inputs_path[i]
    #
    #         file_name = os.path.basename(input_path)  # 获取文件名
    #         class_name = os.path.basename(os.path.dirname(input_path))
    #
    #         target_path = correct_samples_folder  # 目标路径
    #
    #         if 'ori' in input_path and ori_saved<50:
    #             ori_folder = os.path.join(target_path, 'ori')
    #             if not os.path.exists(ori_folder+'/'+class_name):  # 如果子文件夹不存在，创建它
    #                 os.makedirs(ori_folder+'/'+class_name)
    #             target_path = ori_folder+'/'+class_name  # 设置目标路径为'ori'文件夹
    #             ori_saved+=1
    #             if os.path.isfile(input_path):
    #                 shutil.copy(input_path, os.path.join(target_path, file_name))
    #
    #         if 'adv' in input_path and adv_saved<50:
    #             adv_folder = os.path.join(target_path, 'adv')
    #             if not os.path.exists(adv_folder+'/'+class_name):  # 如果子文件夹不存在，创建它
    #                 os.makedirs(adv_folder+'/'+class_name)
    #             target_path = adv_folder+'/'+class_name  # 设置目标路径为'ori'文件夹
    #             adv_saved+=1
    #
    #         # 如果文件存在，复制到新文件夹
    #             if os.path.isfile(input_path):
    #                 shutil.copy(input_path, os.path.join(target_path, file_name))

    return accuracy


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
        "Model": 'AlexNet',
        "Attack_Method": "FGSM-Lambda1"
    }
    start_time = time.time()
    result = detect_main(defparam)
    end_time = time.time()
    total_time=(106.516+0.1093)/(146+346+176+318)
    print(total_time)
    print(result)
