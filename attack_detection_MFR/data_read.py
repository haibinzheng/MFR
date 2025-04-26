import numpy as np
import os

cl_all=[]
for filename in os.listdir('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/cl'):     #namelist
    # cl1=np.load(os.path.join('/home/Bear/attack_detection/cw_resnet_cifar10/orin/cl',filename))
    # # adv_image=np.append(adv_image)
    name0=str(filename).split('.npy')[0]
    name=int(name0)
    cl_all.append(name)
cl_all=np.array(cl_all)
print(cl_all.shape)
print(cl_all)
print(np.min(cl_all))
print(np.max(cl_all))
cl_all=sorted(cl_all)
cl_all=np.array(cl_all)
print(cl_all.shape)
print(cl_all)
print(np.min(cl_all))
print(np.max(cl_all))
np.save('/home/Bear/attack_detection/cw_vgg19_cifar10/orin/cl',cl_all)