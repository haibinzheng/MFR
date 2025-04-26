import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import tqdm

# 定义模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128*128*3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 4096)
        self.fc6 = nn.Linear(4096, 512)
        self.fc7 = nn.Linear(512, 1024)
        self.fc8 = nn.Linear(1024, 128*128*3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = x.view(x.size(0), -1)
        return x

# 路径和参数设置
Dataset_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets_generative'
Model_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/GenerativeAutoEncoder_models'
lable_txt = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets100_Chinese.csv'
batch_size = 64
scales = 20 # 生成图像的扰动大小，值越小，扰动占比越少
N_num = 100 # 设置当前类生成的样本数
epochs = 10 # 设置训练迭代轮数
data_class = ['ZJ', 'TK', 'JC', 'HJ', 'JGQ']
data_class_name = data_class[4] # 设置类别，这里有装甲、坦克、舰船、火箭、机关枪类

# 对数据集进行预处理
transform_imagenet = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: x.view(-1)),
            ])

trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, 'train_' + data_class_name),
            transform=transform_imagenet)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# 实例化模型和损失函数
generator = Generator()
print(generator)

# 定义要剪枝的层和比例
prune_layers = [generator.fc5]
ratio1 = 0.1  # 设置剪枝比例
batch_size = 256

# 剪枝操作
for layer in prune_layers:
    # 假设要移除权重的10%
    amount_to_prune = int(layer.weight.numel() * ratio1)
    print(f'Pruning {amount_to_prune} weights from {layer.weight.shape}')

    # 找到权重中最小的权重
    # loss = nn.L1Loss()
    _, idx = torch.sort(torch.abs(layer.weight.data))
    # threshold_idx = idx[:amount_to_prune]
    # print(idx)

    # 设置这些权重为0
    mask = torch.ones_like(layer.weight.data)
    for i in range(layer.weight.shape[0]):
        for j in range( int(layer.weight.shape[1] *ratio1) ):
            mask[ i ][ idx[i][j] ] = 0
    # print(mask.sum())
    layer.weight.data.mul_(mask)
# 模型剪枝后可以继续使用model进行训练和推理

torch.save(generator.state_dict(), './GenerativeAutoEncoder_models/model_pruning%d%%Rate_'%(ratio1*100)+data_class_name+'.pth')
print("已完成模型的剪枝，剪枝率为%d%%."%(int(ratio1*100)))

# # 模型训练
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
# generator.to(device)
# # 训练模型
# for epoch in range(epochs):
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         noise = torch.randn_like(inputs)
#         noise = noise.to(device)
#         outputs = generator(noise)
#         loss = criterion(outputs, inputs)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (i + 1) % 10 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch + 1, epochs, i + 1, len(trainloader), loss.item()))
#
#     # 保存模型
#     torch.save(generator.state_dict(), './GenerativeAutoEncoder_models/model_'+data_class_name+'.pth')


# # 推理模型生成样本
# generator.load_state_dict(torch.load(Model_dir, '/model_'+data_class_name+'.pth'))
# generator.to(device)
#
# # 随机生成N_num张样本
# for iter in range(N_num):
#     noise = torch.normal(mean=0.0, std=1.0, size=(1, 128*128*3))
#     noise = noise.to(device)
#     outputs = generator(noise)
#     outputs1 = torch.reshape(outputs, [128, 128, 3]).data
#     # outputs1 = torch.reshape(outputs, [3, 128, 128])
#     # outputs1 = torch.transpose(outputs1,0,2).data
#     outputs1 = outputs1.cpu()
#     outputs1 = np.array(outputs1)
#     filenames = os.listdir(os.path.join(Dataset_dir, 'train_' + data_class_name))
#     path_tmp = os.path.join(Dataset_dir, 'train_' + data_class_name, filenames[0])
#     filenames = os.listdir(path_tmp)
#
#     original_image = Image.open(os.path.join(path_tmp, filenames[random.randint(0, len(filenames) - 1)]))
#     original_image = original_image.resize((128, 128))
#     original_image = np.array(original_image)
#     if len(original_image.shape)<3:
#         # 创建一个空的RGB图像
#         rgb_image = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)
#         # 复制灰度图的像素值到RGB图像的三个通道
#         rgb_image[:,:,0]= original_image
#         rgb_image[:,:,1]= original_image
#         rgb_image[:,:,2]= original_image
#         original_image = rgb_image
#
#     save_imgs = original_image + outputs1*scales
#
#     outputs1 = Image.fromarray(save_imgs.astype('uint8'), 'RGB')
#     outputs1.save(os.path.join(Dataset_dir, 'Generated_'+data_class_name, str(iter)+'.JPEG'))




