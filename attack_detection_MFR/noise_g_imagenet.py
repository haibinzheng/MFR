import numpy as np
import skimage
x_noise=[]
x_adnoise=[]




for i in range(1,6):
    noise=np.load('/tmp/fgsm_resnet152_imagenet/noise/noise_gv00{}'.format(str(i))+'m0.npy')
    for j in range(0,4901):
        x_test0 = np.load('/tmp/fgsm_resnet152_imagenet/orin/x_testaa{}'.format(str(j))+'.npy')
        x_adtest0 = np.load('/tmp/fgsm_resnet152_imagenet/orin/x_adtestaa{}'.format(str(j)) + '.npy')

        print(np.max(x_test0))
        print(np.min(x_test0))
        print(np.max(x_adtest0))
        print(np.min(x_adtest0))

        x_testnoise = x_test0 + noise[j]
        x_adtestnoise = x_adtest0 + noise[j]
        #
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

        np.save('/tmp/fgsm_resnet152_imagenet/image_all/x_noise/x_testgv00{}'.format(str(i))+'m0_{}'.format(str(j))+'.npy', x_testnoise)
        np.save('/tmp/fgsm_resnet152_imagenet/image_all/x_noise/x_adtestgv00{}'.format(str(i))+'m0_{}'.format(str(j))+'.npy', x_adtestnoise)