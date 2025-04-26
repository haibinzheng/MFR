import numpy as np
import scipy.misc
# import imagenet_utils as iu

# x_adtest=iu.get_val_dataflow( datadir='/tmp/fgsm_resnet152_imagenet/orin/x_adtest.npy', batch_size=128)
#
# x_test=np.load('/tmp/fgsm_in3v_imagenet/orin/x_test.npy')
# x_adtest=np.load('/tmp/fgsm_in3v_imagenet/orin/x_adtest.npy')


# x_test=np.reshape(x_test,(len(x_test),28,28))
# x_adtest=np.reshape(x_adtest,(len(x_adtest),28,28))
#
# print(x_test.shape)
# print(x_adtest.shape)

for j in [100,150,200,250,280,295,303,350,400,450,500]:
    print('size=', j)
    for i in range(0,4901):
        x_test=np.load('/tmp/fgsm_resnet152_imagenet/orin/x_testaa{}'.format(str(j))+'.npy')
        x_adtest=np.load('/tmp/fgsm_resnet152_imagenet/orin/x_adtestaa{}'.format(str(j))+'.npy')
        x_test=np.reshape(x_test,(299,299,3))
        x_adtest=np.reshape(x_adtest,(299,299,3))

        # print(np.max(x_test))
        # print(np.min(x_test))
        # print(np.max(x_adtest))
        # print(np.min(x_adtest))

        length = len(x_test)
        x_adtest = x_adtest * 255
        x_test = x_test * 255

        # print(np.max(x_test))
        # print(np.min(x_test))
        # print(np.max(x_adtest))
        # print(np.min(x_adtest))
        # print(x_test.shape)
        # print(x_adtest.shape)



        x_testreshape1=scipy.misc.imresize(x_test,(j,j), interp='bilinear')
        x_adtestreshape1=scipy.misc.imresize(x_adtest,(j,j), interp='bilinear')


        print(x_testreshape1.shape)
        print(x_adtestreshape1.shape)


        x_testreshape2=scipy.misc.imresize(x_testreshape1,(299,299), interp='bilinear')
        x_adtestreshape2=scipy.misc.imresize(x_adtestreshape1,(299,299), interp='bilinear')


        print(x_testreshape2.shape)
        print(x_adtestreshape2.shape)

        x_adtestreshape2=x_adtestreshape2/255
        x_testreshape2=x_testreshape2/255

        print(np.max(x_testreshape2))
        print(np.min(x_testreshape2))
        print(np.max(x_adtestreshape2))
        print(np.min(x_adtestreshape2))

        np.save('/tmp/fgsm_resnet152_imagenet/image_all/x_resize/x_testrs{}'.format(str(j)) + '_{}'.format(str(i)) + '.npy', x_testreshape2)
        np.save('/tmp/fgsm_resnet152_imagenet/image_all/x_resize/x_adtestrs{}'.format(str(j)) + '_{}'.format(str(i)) + '.npy', x_adtestreshape2)