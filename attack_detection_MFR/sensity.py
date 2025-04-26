import numpy as np
from sklearn.model_selection import train_test_split

x_56co=np.load('/home/Bear/attack_detection/WorB/X56_co.npy')
x_56ad=np.load('/home/Bear/attack_detection/WorB/X56_ad.npy')

print(x_56co.shape)
print(x_56ad.shape)
y_56co=np.zeros((len(x_56co)))
y_56ad=np.zeros((len(x_56ad)))+1



print(y_56co)
print(np.max(y_56co))
print(np.min(y_56co))

print(y_56ad)
print(np.max(y_56ad))
print(np.min(y_56ad))

X = np.concatenate((x_56co,x_56ad), axis=0)
print(X.shape)
print(np.max(X))
print(np.min(X))
Y = np.concatenate((y_56co,y_56ad), axis=0)
print(Y.shape)
print(np.max(Y))
print(np.min(Y))

np.save('/tmp/sensity/X_56',X)
np.save('/tmp/sensity/Y_56',Y)
print(X.shape)
print(Y.shape)
