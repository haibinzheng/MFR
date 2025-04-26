import torch

import torch.nn as nn
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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

Dataset_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets'
lable_txt = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets100_Chinese.csv'
batch_size = 256
num_classes = 10
epochs = 10
trained_model_name = 'S4_transfer_VGGNet_10Class_62Acc_3Epoch.pth' # 用于迁移学习的模型
transfer_train_data = 'train_transfer50_60class' # 选择迁移的训练数据集
transfer_test_data = 'test_transfer50_60class'

# 对数据集进行预处理
transform_imagenet = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, transfer_train_data),
            transform=transform_imagenet)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.ImageFolder(
    root=os.path.join(Dataset_dir, transfer_test_data),
    transform=transform_imagenet)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False, num_workers=0)
print("训练集样本数:",len(trainset.targets))
print("测试集样本数：",len(testset.targets))

model = MyVggNet(num_classes=num_classes)
#导入预训练好的VGG模型
# model.load_state_dict(torch.load(os.path.join('./VGGNet_models', trained_model_name)))

model.to(device)

# 训练模型
criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
last=0
hostlist_set = []
acc1 = 0
for epoch in range(epochs):
   print('\nEpoch: %d' % (epoch + 1))
   model.train()
   sum_loss = 0.0
   correct = 0.0
   total = 0.0
   for i, data in enumerate(trainloader, 0):
       # prepare dataset
       length = len(trainloader)
       inputs, labels = data
       inputs, labels = inputs.to(device), labels.to(device)
       optimizer.zero_grad()

       # forward & backward
       outputs = model(inputs)
#            outputs,aux2,aux1 = model(inputs)
       loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()

       # print ac & loss in each batch
       sum_loss += loss.item()
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += predicted.eq(labels.data).cpu().sum()
       print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
       state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

   # get the ac with valdataset in each epoch
   print('Waiting Test...')
   with torch.no_grad():
       correct = 0
       total = 0
       num = 0
       for i, data in enumerate(testloader, 0):
           num = num+1
           model.eval()
           images, labels = data
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum()
           test_loss = criterion(outputs, labels)
           sum_loss += test_loss.item()
       # print('Test\'s ac is: %.3f%%' % (100 * torch.true_divide(correct, total)))
           final_loss =round(sum_loss/(i+1),2)
   now_acc =round ((100. * correct / total).item(),2)
   print('Test acc:', now_acc)

   if now_acc>acc1:
       acc1 = now_acc
       # print('VGGNet100_acc%d.pth'%(acc1))
       torch.save(model.state_dict(), './VGGNet_models/S5_transfer_VGGNet_%dClass_%dAcc_%dEpoch.pth'%(num_classes, acc1,epoch))