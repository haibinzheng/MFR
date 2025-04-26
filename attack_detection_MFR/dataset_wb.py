import numpy as np
# x_train=np.load('/home/Bear/attack_detection/WorB/x_train56.npy')
# y_train=np.load('/home/Bear/attack_detection/WorB/y_train56.npy')
# x_test=np.load('/home/Bear/attack_detection/WorB/x_test56.npy')
# y_test=np.load('/home/Bear/attack_detection/WorB/y_test56.npy')
X_ad=np.load('/home/Bear/attack_detection/WorB/X56_ad.npy')
Y_ad=np.load('/home/Bear/attack_detection/WorB/Y56_co.npy')
X_co=np.load('/home/Bear/attack_detection/WorB/X56_co.npy')
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
print(X_ad.shape)
print(Y_ad.shape)
print(X_co.shape)
x_wb_rotate_train=[]
x_wb_rotate_test=[]
x_ad=[]
x_co=[]
# 相似性高
# for i in range(0,len(X_ad)):
#     for cl in [1,18,17,16,19,20,3]:
#         x_ad.append(X_ad[i,cl,:])
# x_ad=np.array(x_ad)
# x_ad=np.reshape(x_ad,(53586,7,10))
# print(x_ad.shape)
#
# for i in range(0,len(X_co)):
#     for cl in [1,18,17,16,19,20,3]:
#         x_co.append(X_co[i,cl,:])
# x_co=np.array(x_co)
# x_co=np.reshape(x_co,(53586,7,10))
# print(x_co.shape)
# print(X_ad[0,19])
# print(x_ad[0,4])
# print(X_co[0,1])
# print(x_co[0,0])
# # np.save('/home/Bear/attack_detection/WorB/X_rotate_5',x)
# X_rotate_5=np.concatenate((x_ad,x_co),axis=1)
# print(X_rotate_5.shape)
# np.save('/home/Bear/attack_detection/WorB/X_rotate_coad7',X_rotate_5)
# np.save('/home/Bear/attack_detection/WorB/Y_rotate_coad7',Y_ad)
# ************************************************************************************************************8


# # 参差不齐cc
for i in range(0,len(X_ad)):
    for cl in [1,18,17,16,19,20,3]:
        x_ad.append(X_ad[i,cl,:])
        x_co.append(X_co[i,cl,:])
x_ad=np.array(x_ad)
x_ad=np.reshape(x_ad,(53586,7,10))
print(x_ad.shape)

x_co=np.array(x_co)
x_co=np.reshape(x_co,(53586,7,10))
print(x_co.shape)
print(X_ad[0,26])
print(x_ad[0,2])
print(X_co[0,46])
print(x_co[0,6])
# np.save('/home/Bear/attack_detection/WorB/X_rotate_5',x)
X_rotate_5=np.concatenate((x_ad,x_co),axis=1)
print(X_rotate_5.shape)

x=[]
for i in range(0,len(X_ad)):
    for cl in [1,18,17,16,19,20,3]:
        x.append(X_co[i, cl, :])
        x.append(X_ad[i, cl,:])

x=np.array(x)
print('x.shape',x.shape)
x=np.reshape(x,(53586,14,10))
print(x.shape)
print(X_co[0,20])
print(x[0,10])
print(X_ad[0,20])
print(x[0,11])
np.save('/home/Bear/attack_detection/WorB/X_cc_coda7',x)
np.save('/home/Bear/attack_detection/WorB/Y_cc_coda7',Y_ad)



# 相减
# X_rotate_5=x_co-x_ad
# print(X_rotate_5.shape)
# np.save('/home/Bear/attack_detection/WorB/X_rotate_de5',X_rotate_5)
# np.save('/home/Bear/attack_detection/WorB/Y_rotate_de5',Y_ad)


