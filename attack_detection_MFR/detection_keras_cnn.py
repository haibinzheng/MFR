import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D,GlobalMaxPooling1D
import os
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


batch_size = 32
num_classes = 2
epochs = 200
# data_augmentation = True
# num_predictions = 20
save_dir = os.path.join(os.getcwd(), '1126_cnn_saved_models')
model_name = 'cifar10_1126s_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

x_train=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/data_0135710gv123z1z2x1/x_train_p.npy')
y_train=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/data_0135710gv123z1z2x1/y_train_p.npy')
x_test=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/data_0135710gv123z1z2x1/x_test_p.npy')
y_test=np.load('/home/Bear/attack_detection/fgsm_cnn_cifar10/data_0135710gv123z1z2x1/y_test_p.npy')
# x_train=np.expand_dims(x_train,axis=3)
print(x_train.shape)

y_train=keras.utils.to_categorical(y_train,2)
y_test=keras.utils.to_categorical(y_test,2)
print(x_train.shape)
# print(x_train[0,0])
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model=Sequential()
# model.add(Conv1D(32,input_shape=x_train.shape[1:],kernel_size=2,strides=1,padding='same',activation='relu'))
# model.add(Flatten())
# model.add(Dense(200,input_shape=x_train.shape,activation='relu'))
model.add(Dense(200,input_dim=12,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax') )



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Training
model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs,verbose=1,callbacks=[reduce_lr],
          validation_data=(x_test,y_test))

score=model.evaluate(x_test,y_test,verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

