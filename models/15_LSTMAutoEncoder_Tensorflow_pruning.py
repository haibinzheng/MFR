import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import numpy as np

from PIL import Image
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.layers import Input, Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization, LSTM, Activation
from keras.models import Model


# 定义模型
# class MyLSTM_AE(tf.keras.Model):
#     def __init__(self):
#         super(MyLSTM_AE, self).__init__()
#         self.encoder_lstm = tf.keras.layers.LSTM(128, return_sequences=True)
#         self.decoder_lstm = tf.keras.layers.LSTM(128, return_sequences=True)
#
#     def call(self, inputs):
#         encoded = self.encoder_lstm(inputs)
#         decoded = self.decoder_lstm(encoded)
#         return decoded

# class MyLSTM_AE(tf.keras.Model):
#     def __init__(self):
#         super(MyLSTM_AE, self).__init__()
#         self.encoder_conv = tf.keras.Sequential([
#             layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#             layers.MaxPooling2D((2, 2), padding='same'),
#             layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#             layers.MaxPooling2D((2, 2), padding='same')
#         ])
#         self.encoder_lstm = tf.keras.layers.LSTM(128, return_sequences=True)
#         self.decoder_lstm = tf.keras.layers.LSTM(128, return_sequences=True)
#         self.decoder_conv = tf.keras.Sequential([
#             layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#             layers.UpSampling2D((2, 2)),
#             layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#             layers.UpSampling2D((2, 2)),
#             layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
#         ])
#
#     def call(self, inputs):
#         encoded = self.encoder_conv(inputs)
#         encoded = tf.keras.layers.Reshape((-1, np.prod(encoded.shape[1:])))(encoded)
#         encoded = self.encoder_lstm(encoded)
#         decoded = self.decoder_lstm(encoded)
#         decoded = tf.keras.layers.Reshape((-1, 8, 8, 64))(decoded)
#         decoded = self.decoder_conv(decoded)
#         return decoded

def MyLSTM_AE(inshape=(1,128*128*3), class_num=128*128*3, lr=1e-4):
    input = Input(shape=inshape)
    # output = LSTM(units=127, activation="tanh", return_sequences=True)(input)
    output = LSTM(units=256, activation="tanh", return_sequences=True)(input)
    output = LSTM(units=64, activation="tanh", return_sequences=True)(output)

    output = Flatten()(output)
    output = Dense(1024, activation=None)(output)
    output = BatchNormalization()(output)
    output = Activation(activation="relu")(output)
    output = Dropout(0.5)(output)

    # output = Dense(128, activation=None)(output)
    # output = BatchNormalization()(output)
    # output = Activation(activation="relu")(output)

    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)

    output1 = Dense(class_num, activation='relu')(output)
    # output2 = Dense(class_num, activation='linear')(output)
    model = Model(input, output1)
    # optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer='adam', loss="mse", metrics=["mae"])
    # model.summary()
    return model

# 读取图像数据
def read_image_folder(folder_path, image_size, dim = 1):
    images = []
    for filename in os.listdir(folder_path):
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(folder_path, filename),
            target_size=image_size
        )
        img = tf.keras.preprocessing.image.img_to_array(img)
        if dim == 2:
            img = np.reshape(img, (1, image_size[0]*image_size[1]*image_size[2]))
        if dim == 1:
            img = np.reshape(img, (image_size[0]*image_size[1]*image_size[2]))
        images.append(img)
    print(np.shape(np.array(images)))
    return np.array(images)

# 定义参数
input_shape = (1,128*128*3)
image_size = (128,128,3)
batch_size = 32
epochs = 10
learning_rate = 0.001
scales = 20 # 生成图像的扰动大小，值越小，扰动占比越少

# 构建模型
model = MyLSTM_AE()
model.build(input_shape=input_shape)
model.compile(optimizer='adam', loss='mse')

Dataset_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets_generative_LSTM'
Model_dir = '/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/LSTM_models/'

data_class = ['ZJ', 'TK', 'JC', 'HJ', 'JGQ']
class_number = 3
data_class_name = data_class[class_number] # 设置类别，这里有装甲、坦克、舰船、火箭、机关枪类
path_class = [
    "/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets_generative_LSTM/train_ZJ/n03478589",
    "/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets_generative_LSTM/train_TK/n04389033",
    "/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets_generative_LSTM/train_JC/n02687172",
    "/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets_generative_LSTM/train_HJ/n04266014",
    "/data0/BigPlatform/haibinzheng/OutSourcing/ModelProject_ZJUT/Datasets/Datasets_generative_LSTM/train_JGQ/n04090263"
]


# 从文件夹读取图像数据并处理成训练数据
input_folder = path_class[class_number]

input_data = read_image_folder(input_folder, image_size, dim = 2)
# print(input_data[0])
input_data = np.random.rand(*input_data.shape)

output_data = read_image_folder(input_folder, image_size, dim = 1)


# input_data = output_data + 0.5 * np.random.normal(size=output_data.shape)  # 添加高斯噪声


model_path = os.path.join(Model_dir, "LSTM_AutoEncoder_"+ data_class[class_number] +".h5")
model.load_weights(model_path)

# 定义要剪枝的层和比例
layer_to_prune = model.layers[1]
ratio1 = 0.1  # 设置剪枝比例

# 获取该层的权重
weights = layer_to_prune.get_weights()

# 对权重进行剪枝操作，比如将部分权重设置为零
pruned_weights = []
for w in weights:
    if np.random.rand() > ratio1:
        pruned_weights.append(w)
    else:
        pruned_weights.append(tf.zeros_like(w))

# 将剪枝后的权重设置回模型中的对应层
layer_to_prune.set_weights(pruned_weights)

model.save('./LSTM_models/model_pruning%d%%Rate_'%(ratio1*100)+data_class_name+'.h5')
print("已完成模型的剪枝，剪枝率为%d%%."%(int(ratio1*100)))
print("模型保存路径：%s"%(Model_dir))