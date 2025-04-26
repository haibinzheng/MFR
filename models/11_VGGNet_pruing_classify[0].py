import torch
import torch.nn as nn
import torch.nn as nn
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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

# 实例化模型
model = MyVggNet()
# print(model)
Dataset_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets'
Model_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/VGGNet_models'

trained_model_name = 'S10_transfer_VGGNet_10Class_62Acc_5Epoch.pth' # 用于剪枝的模型
pruning_test_data = 'test_transfer00_10class'

model.load_state_dict(torch.load(os.path.join(Model_dir, trained_model_name)))

# 定义要剪枝的层和比例
prune_layers = [model.classify[0]]
ratio1 = 0.9  # 设置剪枝比例
batch_size = 256
num_classes = 10

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

# 对数据集进行预处理
transform_imagenet = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

testset = torchvision.datasets.ImageFolder(
    root=os.path.join(Dataset_dir, pruning_test_data),
    transform=transform_imagenet)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False, num_workers=0)
print("测试集样本数：",len(testset.targets))


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
print("剪枝后模型的测试准确率：%.2f%%" % (acc_tmp))

# 保存模型
torch.save(model.state_dict(), './VGGNet_models/S11_pruning_VGGNet_%dClass_%dAcc_%d%%Ratio.pth'%(num_classes, acc_tmp, ratio1*100))


