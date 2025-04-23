from nets.alexnet import AlexNet_v1, AlexNet_v2
from util_kd import *
import numpy as np
from sklearn.neighbors import KernelDensity
import glob
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import copy
import csv
import json
import time

BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00, 'imagenet': 0.15, 'military': 0.45, 'warship': 1.00}
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
        if dataset_name == 'warship':
            Y_train_onehot = one_hot_encode(Y_train, 2)
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
    best_acc = 0
    Model = params['Model']
    attack_method = params['Attack_Method']

    dataset_name = 'military'

    batch_size = 1

    if Model == 'AlexNet':
        model = AlexNet_v1(num_classes=10)
        model.load_weights(
            '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image_tf/save_model/military/alexnet_0.6307123899459839.ckpt')
    print("当前测试的攻击方法是: " + attack_method)
    model.summary()
    model.trainable = False

    class_to_label = {}

    if dataset_name == 'imagenet':
        test_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/imagenet_100/test'
        train_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/imagenet_100/train'
    elif dataset_name == 'military':
        test_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/Military/test'
        train_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/clean_datasets/Military/train'
    elif dataset_name == 'warship':
        test_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/warship/test'
        train_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image/datasets/warship/train'

        # 生成类名到数字标签的映射
        # 除了MILITARY请使用以下代码  MILITARY类别标签需要靠外部文件
    # for i, class_folder in enumerate(sorted(glob.glob(os.path.join(test_dataset_dir, "*")))):
    #     class_name = os.path.basename(class_folder)
    #     class_to_label[class_name] = i  # 标签映射
    # 非MILITARY请注释以下代码
    with open("/data/Newdisk/chenjingwen/qkh/AlexNet_TensorFlow/class_indices_military.json", 'r',
              encoding='UTF-8') as f:
        class_to_label = json.load(f)
        class_to_label = dict(zip(class_to_label.values(), class_to_label.keys()))
        for key in class_to_label.keys():
            class_to_label[key] = int(class_to_label[key])
    ###############
    print(class_to_label)
    X_test = []
    Y_test = []

    # 遍历 DataLoader
    for class_folder in glob.glob(os.path.join(test_dataset_dir, "*")):
        class_name = class_to_label[os.path.basename(class_folder)]
        class_images = glob.glob(os.path.join(class_folder, "*.JPEG"))
        for class_image in class_images:
            img = Image.open(class_image)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((im_width, im_height))
            img = np.array(img) / 255.
            img = np.expand_dims(img, 0)
            X_test.append(img)
            Y_test.append(class_name)

    adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image_tf/dataset/military/AlexNet/FGSM_less_less/adv'
    # adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/attack_datasets/AlexNet_data/{}All/adv'.format(
    #         attack_type)

    X_adv = []
    Y_adv = []
    adv_images = []
    for class_folder in glob.glob(os.path.join(adv_dataset_dir, "*")):
        class_name = os.path.basename(class_folder)
        class_images = glob.glob(os.path.join(class_folder, "*.JPEG"))

        for class_image in class_images:
            img = Image.open(class_image)
            adv_images.append(class_image)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((im_width, im_height))
            img = np.array(img) / 255.
            img = np.expand_dims(img, 0)
            X_adv.append(img)
            Y_adv.append(class_name)

    ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image_tf/dataset/military/AlexNet/FGSM_less_less/ori'
    # ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/attack_datasets/AlexNet_data/{}All/ori'.format(
    #     attack_type)

    X_ori = []
    Y_ori = []
    ori_images = []
    for class_folder in glob.glob(os.path.join(ori_dataset_dir, "*")):
        class_name = os.path.basename(class_folder)
        class_images = glob.glob(os.path.join(class_folder, "*.JPEG"))

        for class_image in class_images:
            img = Image.open(class_image)
            ori_images.append(class_image)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((im_width, im_height))
            img = np.array(img) / 255.
            img = np.expand_dims(img, 0)
            X_ori.append(img)
            Y_ori.append(class_name)

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

    ori_input_paths = []
    adv_input_paths = []
    csv_path = 'features_{}.csv'.format(dataset_name)

    for i in range(len(ori_images)):
        ori_image = ori_images[i]
        input = Image.open(ori_image)
        if input.mode != 'RGB':
            input = input.convert('RGB')
        input = input.resize((im_width, im_height))
        input = np.array(input) / 255.
        input = np.expand_dims(input, 0)

        X_test_ori_features = get_deep_representations(model, input)
        result = np.squeeze(model.predict(input))
        preds_test_ori = np.argmax(result)
        densities_ori = score_samples(
            kdes,
            X_test_ori_features,
            preds_test_ori
        )

        # uncertainty_mean=get_bayesian_uncertainty(Model,input)
        # inconsistency_mean = get_bayesian_inconsistency(Model,input)
        #
        # adv_inconsistency_mean.append(inconsistency_mean)
        # adv_uncertainty_mean.append(uncertainty_mean)

        clean_kd.append(densities_ori)
        clean_label.append(0)

    for i in range(len(adv_images)):
        adv_image = adv_images[i]
        input = Image.open(adv_image)
        if input.mode != 'RGB':
            input = input.convert('RGB')
        input = input.resize((im_width, im_height))
        input = np.array(input) / 255.
        input = np.expand_dims(input, 0)

        X_test_adv_features = get_deep_representations(model, input)
        result = np.squeeze(model.predict(input))
        preds_test_adv = np.argmax(result)
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

        adv_kd.append(densities_adv)
        adv_label.append(1)

    header = ['label', 'input', 'kd']
    datarows = []
    for kd, input, label in zip(adv_kd, adv_images, adv_label):
        datarows.append([label, input, kd])
    for kd, input, label in zip(clean_kd, ori_images, clean_label):
        datarows.append([label, input, kd])
    write_csv(csv_path, header, datarows)

    # X = np.array(adv_kd + clean_kd).reshape(-1, 1)  # 将数据转换为二维数组，符合sklearn的要求
    #
    # y = np.array(adv_label + clean_label)
    #
    # # 划分数据集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # print("使用三者结合进行预测：")
    # model_filename = 'kd_save/save_rf/rf_{}_{}.pkl'.format(Model, attack_type)
    # if os.path.isfile(model_filename):
    #     print("分类模型已经存在")
    #     classifier_2 = load(model_filename)
    # else:
    #     print("分类模型路径不存在，重新训练一个")
    #     classifier_2 = RandomForestClassifier(n_estimators=100)
    #     classifier_2.fit(X_train, y_train)
    #     # dump(classifier_2, model_filename)
    #
    # y_pred = classifier_2.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    #
    # all_result=[]
    # all_result.append(accuracy)
    # cm = confusion_matrix(y_test, y_pred)
    #
    # print("confusion Matrix: ")
    # print(cm)
    # print("Accuracy with Random Forest:", accuracy)
    #
    # print(attack_method + '的最终ACC是：' + str(sum(all_result) / len(all_result)))
    #
    jsontext = {
        'Message': '相关特征已写入csv'

    }

    return jsontext


#
# def detect(params):
#     best_acc = 0
#     Model = params['Model']
#     attack_method = params['Attack_Method']
#
#     if Model in ['AlexNet', 'GoogleNet']:
#         dataset_name = 'imagenet'
#     else:
#         dataset_name = 'military'
#
#     if attack_method in ['FGSM-Lambda1', 'DeepFool-Lambda1', 'PatchAttack', 'NFA-Lambda1', 'MIFGSM-Lambda1',
#                          'JSMA-Lambda1', 'FGSM-L1', 'EAD-Lambda1', 'Adef-Lambda1', 'PGD-Lambda1',
#                          'CarliniWagnerL2Attack']:
#         attack_type = 'WhiteBox'
#
#     else:
#         attack_type = 'BlackBox'
#
#     batch_size = 1
#     if Model == 'AlexNet':
#         model = AlexNet_v1(num_classes=100)
#         model.load_weights(
#             '/data/Newdisk/chenjingwen/DT_B4/SJS_detect/Image_tf/save_model/military/alexnet_0.6307123899459839.ckpt')
#
#     print("当前测试的攻击方法是: " + attack_method)
#     adv_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/selected_attack_datasets/{}/{}/adv_png'.format(
#         Model,
#         attack_method)
#
#     ori_dataset_dir = '/data/Newdisk/chenjingwen/DT_B4/GD_detect/image/selected_attack_datasets/{}/{}/ori_png'.format(
#         Model,
#         attack_method)
#
#     kdes = get_kd(Model, model, [], [], [], [], dataset_name)
#
#     result = []
#
#     adv_uncertainty_mean = []
#     adv_inconsistency_mean = []
#     adv_kd = []
#     adv_label = []
#     clean_uncertainty_mean = []
#     clean_inconsistency_mean = []
#     clean_kd = []
#     clean_label = []


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
    print(end_time - start_time)
    print(result)
