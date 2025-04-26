import numpy as np
X=np.load('/tmp/sensity/X_56.npy')
Y=np.load('/tmp/sensity/Y_56.npy')


print(X.shape)
print(Y.shape)

x=[]
# 相似性高
for i in range(0,len(X)):
    for cl in [0,16,20,15,5]:
        x.append(X[i,cl,:])
x=np.array(x)
print(x.shape)
x=np.reshape(x,(107172,5,10))
print(x.shape)
print(X[0,15])
print(x[0,3])

np.save('/tmp/sensity/X_rs44',x)
np.save('/tmp/sensity/Y_rs44',Y)
# ************************************************************************************************************8





