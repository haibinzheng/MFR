import numpy as np
import skimage

x_test0 = np.load('/tmp/fgsm_resnet152_imagenet/orin1/x_test2000.npy')
x_adtest0 = np.load('/tmp/fgsm_resnet152_imagenet/orin1/x_adtest2000.npy')

print(np.max(x_test0))
print(np.min(x_test0))
print(np.max(x_adtest0))
print(np.min(x_adtest0))

for i in range(1,6):
    x_testnoise=[]
    x_adtestnoise=[]
    print('noise=',i*0.01)
    x_noise=skimage.util.random_noise(x_test0,mode='gaussian',var=0.01*i,mean=0, seed=None, clip=False)
    # print(np.max(x_noise))
    # print(np.min(x_noise))
    noise=x_noise-x_test0
    print(np.max(noise))
    print(np.min(noise))
    # np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/noise/noise_gv00{}'.format(str(i))+'m0.npy', noise)
    x_testnoise = x_test0 + noise
    x_adtestnoise = x_adtest0 + noise

    print(np.max(x_testnoise))
    print(np.min(x_testnoise))
    print(np.max(x_adtestnoise))
    print(np.min(x_adtestnoise))
    x_testnoise = np.clip(x_testnoise, 0, 1)
    x_adtestnoise = np.clip(x_adtestnoise, 0, 1)
    print(np.max(x_testnoise))
    print(np.min(x_testnoise))
    print(np.max(x_adtestnoise))
    print(np.min(x_adtestnoise))

    print(x_testnoise.shape)
    print(x_adtestnoise.shape)
    np.save('/tmp/fgsm_resnet152_imagenet/image_all/all_2000/x_testgv00{}'.format(str(i))+'m0.npy', x_testnoise)
    np.save('/tmp/fgsm_resnet152_imagenet/image_all/all_2000/x_adtestgv00{}'.format(str(i))+'m0.npy', x_adtestnoise)