from __future__ import print_function
import numpy as np
from PIL import Image
x_test=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/orin/x_test8129.npy')
x_adtest=np.load('/home/Bear/attack_detection/fgsm_vgg16_cifar10/orin/x_adtest8129.npy')
print(x_test.shape)
print(x_adtest.shape)
length=len(x_test)

for j in ['s1']:
    if j=='s1':
        zero = np.zeros((length, 1, 32, 3))
        print(zero.shape)

        sis1=[]
        sis2=[]

        x_test_ts = np.concatenate((x_test, zero), axis=1)
        print(x_test_ts.shape)
        x_adtest_ts = np.concatenate((x_adtest, zero), axis=1)
        print(x_adtest_ts.shape)

        x_test_ts1 = []
        for i in range(0, length):
            sis1 = np.delete(x_test_ts[i], 0, axis=0)
            # sis2 = np.delete(sis1,-1, axis=0)
            x_test_ts1.append(sis1)
            # print(sis2.shape)
        x_test_ts1 = np.array(x_test_ts1)
        print('x_test_ts1.shape', x_test_ts1.shape)
        np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_tests1', x_test_ts1)

        x_adtest_ts1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_ts[i], 0, axis=0)
            # sis2 = np.delete(sis1,-1, axis=0)
            x_adtest_ts1.append(sis1)
            # print(sis2.shape)
        x_adtest_ts1 = np.array(x_adtest_ts1)
        print('x_adtest_ts1.shape', x_adtest_ts1.shape)
        np.save('/home/Bear/attack_detection/fgsm_vgg16_cifar10/cifar10_all/x_adtests1', x_adtest_ts1)

    elif j=='s2':
        zero = np.zeros((length, 2, 32, 3))
        print(zero.shape)

        sis1=[]
        sis2=[]

        x_test_ts = np.concatenate((x_test, zero), axis=1)
        print(x_test_ts.shape)
        x_adtest_ts = np.concatenate((x_adtest, zero), axis=1)
        print(x_adtest_ts.shape)

        x_test_ts1 = []
        for i in range(0, length):
            sis1 = np.delete(x_test_ts[i], 0, axis=0)
            sis2 = np.delete(sis1, 0, axis=0)
            x_test_ts1.append(sis2)
            # print(sis2.shape)
        x_test_ts1 = np.array(x_test_ts1)
        print('x_test_ts1.shape', x_test_ts1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_tests2', x_test_ts1)

        x_adtest_ts1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_ts[i], 0, axis=0)
            sis2 = np.delete(sis1, 0, axis=0)
            x_adtest_ts1.append(sis2)
            # print(sis2.shape)
        x_adtest_ts1 = np.array(x_adtest_ts1)
        print('x_adtest_ts1.shape', x_adtest_ts1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_adtests2', x_adtest_ts1)

    elif j=='x1':
        zero = np.zeros((length, 1, 32, 3))
        print(zero.shape)

        sis1=[]
        sis2=[]

        x_test_ts = np.concatenate((zero, x_test), axis=1)
        print(x_test_ts.shape)
        x_adtest_ts = np.concatenate((zero, x_adtest), axis=1)
        print(x_adtest_ts.shape)

        x_test_ts1 = []
        for i in range(0, length):
            sis1 = np.delete(x_test_ts[i], -1, axis=0)
            # sis2 = np.delete(sis1,-1, axis=0)
            x_test_ts1.append(sis1)
            # print(sis2.shape)
        x_test_ts1 = np.array(x_test_ts1)
        print('x_test_ts1.shape', x_test_ts1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_testx1', x_test_ts1)

        x_adtest_ts1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_ts[i], -1, axis=0)
            # sis2 = np.delete(sis1,-1, axis=0)
            x_adtest_ts1.append(sis1)
            # print(sis2.shape)
        x_adtest_ts1 = np.array(x_adtest_ts1)
        print('x_adtest_ts1.shape', x_adtest_ts1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_adtestx1', x_adtest_ts1)

    elif j=='x2':
        m = len(x_test)
        zero = np.zeros((m, 2, 32, 3))
        print(zero.shape)

        sis1=[]
        sis2=[]

        x_test_ts = np.concatenate((zero, x_test), axis=1)
        print(x_test_ts.shape)
        x_adtest_ts = np.concatenate((zero, x_adtest), axis=1)
        print(x_adtest_ts.shape)

        x_test_ts1 = []
        for i in range(0, m):
            sis1 = np.delete(x_test_ts[i], -1, axis=0)
            sis2 = np.delete(sis1, -1, axis=0)
            x_test_ts1.append(sis2)
            # print(sis2.shape)
        x_test_ts1 = np.array(x_test_ts1)
        print('x_test_ts1.shape', x_test_ts1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_testx2', x_test_ts1)

        x_adtest_ts1 = []
        for i in range(0, m):
            sis1 = np.delete(x_adtest_ts[i], -1, axis=0)
            sis2 = np.delete(sis1, -1, axis=0)
            x_adtest_ts1.append(sis2)
            # print(sis2.shape)
        x_adtest_ts1 = np.array(x_adtest_ts1)
        print('x_adtest_ts1.shape', x_adtest_ts1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_adtestx2', x_adtest_ts1)

    elif j=='z1':
        length = len(x_test)
        zero = np.zeros((length, 32, 1, 3))
        print(zero.shape)

        sis1=[]
        sis2=[]

        x_test_z1 = np.concatenate((x_test, zero), axis=2)
        print(x_test_z1.shape)
        x_adtest_z1 = np.concatenate((x_adtest, zero), axis=2)
        print(x_adtest_z1.shape)

        x_test_z1_1 = []
        for i in range(0, length):
            sis1 = np.delete(x_test_z1[i], 0, axis=1)
            # sis2 = np.delete(sis1,0, axis=1)
            x_test_z1_1.append(sis1)
            # print(sis2.shape)
        x_test_z1_1 = np.array(x_test_z1_1)
        print('x_test_z1_1.shape', x_test_z1_1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_testz1', x_test_z1_1)

        x_adtest_z1_1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_z1[i], 0, axis=1)
            # sis2 = np.delete(sis1,0, axis=1)
            x_adtest_z1_1.append(sis1)
            # print(sis2.shape)
        x_adtest_z1_1 = np.array(x_adtest_z1_1)
        print('x_adtest_z1_1.shape', x_adtest_z1_1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_adtestz1', x_adtest_z1_1)

    elif j=='z2':
        length = len(x_test)

        zero = np.zeros((length, 32, 2, 3))
        print(zero.shape)

        sis1=[]
        sis2=[]

        x_test_z1 = np.concatenate((x_test, zero), axis=2)
        print(x_test_z1.shape)
        x_adtest_z1 = np.concatenate((x_adtest, zero), axis=2)
        print(x_adtest_z1.shape)

        x_test_z1_1 = []
        for i in range(0, length):
            sis1 = np.delete(x_test_z1[i], 0, axis=1)
            sis2 = np.delete(sis1, 0, axis=1)
            x_test_z1_1.append(sis2)
            # print(sis2.shape)
        x_test_z1_1 = np.array(x_test_z1_1)
        print('x_test_z1_1.shape', x_test_z1_1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_testz2', x_test_z1_1)

        x_adtest_z1_1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_z1[i], 0, axis=1)
            sis2 = np.delete(sis1, 0, axis=1)
            x_adtest_z1_1.append(sis2)
            # print(sis2.shape)
        x_adtest_z1_1 = np.array(x_adtest_z1_1)
        print('x_adtest_z1_1.shape', x_adtest_z1_1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_adtestz2', x_adtest_z1_1)

    elif j=='y1':
        m = len(x_test)

        zero = np.zeros((m, 32, 1, 3))
        print(zero.shape)
        sis1=[]
        sis2=[]

        x_test_z1 = np.concatenate((zero, x_test), axis=2)
        print(x_test_z1.shape)
        x_adtest_z1 = np.concatenate((zero, x_adtest), axis=2)
        print(x_adtest_z1.shape)

        x_test_z1_1 = []
        for i in range(0, m):
            sis1 = np.delete(x_test_z1[i], -1, axis=1)
            # sis2 = np.delete(sis1,0, axis=1)
            x_test_z1_1.append(sis1)
            # print(sis2.shape)
        x_test_z1_1 = np.array(x_test_z1_1)
        print('x_test_z1_1.shape', x_test_z1_1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_testy1', x_test_z1_1)

        x_adtest_z1_1 = []
        for i in range(0, m):
            sis1 = np.delete(x_adtest_z1[i], -1, axis=1)
            # sis2 = np.delete(sis1,0, axis=1)
            x_adtest_z1_1.append(sis1)
            # print(sis2.shape)
        x_adtest_z1_1 = np.array(x_adtest_z1_1)
        print('x_adtest_z1_1.shape', x_adtest_z1_1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_adtesty1', x_adtest_z1_1)

    elif j=='y2':
        m = len(x_test)
        zero = np.zeros((m, 32, 2, 3))
        print(zero.shape)

        sis1=[]
        sis2=[]

        x_test_z1 = np.concatenate((zero, x_test), axis=2)
        print(x_test_z1.shape)
        x_adtest_z1 = np.concatenate((zero, x_adtest), axis=2)
        print(x_adtest_z1.shape)

        x_test_z1_1 = []
        for i in range(0, m):
            sis1 = np.delete(x_test_z1[i], -1, axis=1)
            sis2 = np.delete(sis1, -1, axis=1)
            x_test_z1_1.append(sis2)
            # print(sis2.shape)
        x_test_z1_1 = np.array(x_test_z1_1)
        print('x_test_z1_1.shape', x_test_z1_1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_testy2', x_test_z1_1)

        x_adtest_z1_1 = []
        for i in range(0, m):
            sis1 = np.delete(x_adtest_z1[i], -1, axis=1)
            sis2 = np.delete(sis1, -1, axis=1)
            x_adtest_z1_1.append(sis2)
            # print(sis2.shape)
        x_adtest_z1_1 = np.array(x_adtest_z1_1)
        print('x_adtest_z1_1.shape', x_adtest_z1_1.shape)
        np.save('/home/Bear/attack_detection/CRA_resnet_cifar10/cifar10_all/x_adtesty2', x_adtest_z1_1)



