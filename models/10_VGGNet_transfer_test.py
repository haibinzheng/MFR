import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#导入预训练好的VGG16
vgg16=models.vgg16(pretrained=False)
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

Dataset_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets'
lable_txt = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets100_Chinese.csv'
batch_size = 256
num_classes = 10
epochs = 2 # 迁移训练十分快，只需要很少个epoch
trained_model_name = 'S10_transfer_VGGNet_10Class_62Acc_5Epoch.pth' # 训练好的迁移学习模型

transfer_test_data = 'test_transfer00_10class'

# 对数据集进行预处理
transform_imagenet = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

testset = torchvision.datasets.ImageFolder(
    root=os.path.join(Dataset_dir, transfer_test_data),
    transform=transform_imagenet)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False, num_workers=0)
print("测试集样本数：",len(testset.targets))

model = MyVggNet(num_classes=num_classes)
model.load_state_dict(torch.load(os.path.join('./VGGNet_models', trained_model_name)))

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
print("迁移学习模型的测试准确率：%.2f%%" % (acc_tmp))
