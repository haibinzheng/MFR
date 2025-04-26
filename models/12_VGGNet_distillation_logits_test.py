import torch
import torch.nn as nn
import torch.nn as nn
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#导入预训练好的VGG16
vgg16=models.vgg16(pretrained=True)
vgg=vgg16.features#获取vgg16的特征提取层
for param in vgg.parameters():
    param.requires_grad = False

class MyVggNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyVggNet, self).__init__()
        # 预训练的vgg16特征提取层
        self.vgg = vgg
        # 添加新的全连接层
        self.classify = nn.Sequential(
            nn.Linear(7*7*512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    # 定义网络的前向传播
    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)  # 多维度的tensor展平成一维
        output = self.classify(x)
        return output

class MyNewSmallNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyNewSmallNet, self).__init__()
        # 预训练的vgg16特征提取层
        self.vgg = vgg
        # 添加新的全连接层
        self.classify = nn.Sequential(
            nn.Linear(7*7*512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    # 定义网络的前向传播
    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)  # 多维度的tensor展平成一维
        output = self.classify(x)
        return output

Dataset_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets'
Model_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/VGGNet_models'
lable_txt = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets100_Chinese.csv'

distilled_model_name = 'S12_distillation_VGGNet_10Class_82Acc60T.pth' # 蒸馏的模型

distillation_train_data = 'train_transfer10_20class'
distillation_test_data = 'test_transfer10_20class'

batch_size = 128
num_classes = 10
epochs = 10

# 定义蒸馏温度T
T = 0.0 # 温度越高，从教师模型中蒸馏的信息越多，蒸馏温度取值【0，1】

# 对数据集进行预处理
transform_imagenet = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

# trainset = torchvision.datasets.ImageFolder(
#             root=os.path.join(Dataset_dir, distillation_train_data),
#             transform=transform_imagenet)
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=batch_size,
#                                           shuffle=True, num_workers=0)
testset = torchvision.datasets.ImageFolder(
    root=os.path.join(Dataset_dir, distillation_test_data),
    transform=transform_imagenet)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False, num_workers=0)
# print("训练集样本数:",len(trainset.targets))
print("测试集样本数：",len(testset.targets))

# 实例化模型,载入被蒸馏的教师模型
model = MyNewSmallNet()

model.load_state_dict(torch.load(os.path.join(Model_dir, distilled_model_name)))

# Test the model
correct = 0
total = 0
model.eval()
model.to(device)
for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

acc_tmp = (correct/total).item()*100
print("蒸馏后模型的测试准确率：%.2f%%" % (acc_tmp))
