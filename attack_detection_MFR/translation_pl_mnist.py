from __future__ import print_function
import numpy as np
from PIL import Image
x_test=np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/x_test00.npy')
x_adtest=np.load('/home/Bear/attack_detection/fgsm_mlp_mnist/orin/x_adtest00.npy')
x_test=np.reshape(x_test,(len(x_test),28,28,1))
x_adtest=np.reshape(x_adtest,(len(x_test),28,28,1))
print(x_test.shape)
print(x_adtest.shape)
np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/x_test00',x_test)
np.save('/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all/x_adtest00',x_adtest)
length=len(x_test)
address='/home/Bear/attack_detection/fgsm_mlp_mnist/mnist_all'
for j in ['s1','s2','x1','x2','z1','z2','y1','y2']:
    if j=='s1':
        zero = np.zeros((length, 1, 28, 1))
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
        np.save('{}'.format(address)+'/x_tests1', x_test_ts1)

        x_adtest_ts1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_ts[i], 0, axis=0)
            # sis2 = np.delete(sis1,-1, axis=0)
            x_adtest_ts1.append(sis1)
            # print(sis2.shape)
        x_adtest_ts1 = np.array(x_adtest_ts1)
        print('x_adtest_ts1.shape', x_adtest_ts1.shape)
        np.save('{}'.format(address)+'/x_adtests1', x_adtest_ts1)

    elif j=='s2':
        zero = np.zeros((length, 2, 28, 1))
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
        np.save('{}'.format(address)+'/x_tests2', x_test_ts1)

        x_adtest_ts1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_ts[i], 0, axis=0)
            sis2 = np.delete(sis1, 0, axis=0)
            x_adtest_ts1.append(sis2)
            # print(sis2.shape)
        x_adtest_ts1 = np.array(x_adtest_ts1)
        print('x_adtest_ts1.shape', x_adtest_ts1.shape)
        np.save('{}'.format(address)+'/x_adtests2', x_adtest_ts1)

    elif j=='x1':
        zero = np.zeros((length, 1, 28, 1))
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
        np.save('{}'.format(address)+'/x_testx1', x_test_ts1)

        x_adtest_ts1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_ts[i], -1, axis=0)
            # sis2 = np.delete(sis1,-1, axis=0)
            x_adtest_ts1.append(sis1)
            # print(sis2.shape)
        x_adtest_ts1 = np.array(x_adtest_ts1)
        print('x_adtest_ts1.shape', x_adtest_ts1.shape)
        np.save('{}'.format(address)+'/x_adtestx1', x_adtest_ts1)

    elif j=='x2':
        m = len(x_test)
        zero = np.zeros((m, 2, 28, 1))
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
        np.save('{}'.format(address)+'/x_testx2', x_test_ts1)

        x_adtest_ts1 = []
        for i in range(0, m):
            sis1 = np.delete(x_adtest_ts[i], -1, axis=0)
            sis2 = np.delete(sis1, -1, axis=0)
            x_adtest_ts1.append(sis2)
            # print(sis2.shape)
        x_adtest_ts1 = np.array(x_adtest_ts1)
        print('x_adtest_ts1.shape', x_adtest_ts1.shape)
        np.save('{}'.format(address)+'/x_adtestx2', x_adtest_ts1)

    elif j=='z1':
        length = len(x_test)
        zero = np.zeros((length, 28, 1, 1))
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
        np.save('{}'.format(address)+'/x_testz1', x_test_z1_1)

        x_adtest_z1_1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_z1[i], 0, axis=1)
            # sis2 = np.delete(sis1,0, axis=1)
            x_adtest_z1_1.append(sis1)
            # print(sis2.shape)
        x_adtest_z1_1 = np.array(x_adtest_z1_1)
        print('x_adtest_z1_1.shape', x_adtest_z1_1.shape)
        np.save('{}'.format(address)+'/x_adtestz1', x_adtest_z1_1)

    elif j=='z2':
        length = len(x_test)

        zero = np.zeros((length, 28, 2, 1))
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
        np.save('{}'.format(address)+'/x_testz2', x_test_z1_1)

        x_adtest_z1_1 = []
        for i in range(0, length):
            sis1 = np.delete(x_adtest_z1[i], 0, axis=1)
            sis2 = np.delete(sis1, 0, axis=1)
            x_adtest_z1_1.append(sis2)
            # print(sis2.shape)
        x_adtest_z1_1 = np.array(x_adtest_z1_1)
        print('x_adtest_z1_1.shape', x_adtest_z1_1.shape)
        np.save('{}'.format(address)+'/x_adtestz2', x_adtest_z1_1)

    elif j=='y1':
        m = len(x_test)

        zero = np.zeros((m, 28, 1, 1))
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
        np.save('{}'.format(address)+'/x_testy1', x_test_z1_1)

        x_adtest_z1_1 = []
        for i in range(0, m):
            sis1 = np.delete(x_adtest_z1[i], -1, axis=1)
            # sis2 = np.delete(sis1,0, axis=1)
            x_adtest_z1_1.append(sis1)
            # print(sis2.shape)
        x_adtest_z1_1 = np.array(x_adtest_z1_1)
        print('x_adtest_z1_1.shape', x_adtest_z1_1.shape)
        np.save('{}'.format(address)+'/x_adtesty1', x_adtest_z1_1)

    elif j=='y2':
        m = len(x_test)
        zero = np.zeros((m, 28, 2, 1))
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
        np.save('{}'.format(address)+'/x_testy2', x_test_z1_1)

        x_adtest_z1_1 = []
        for i in range(0, m):
            sis1 = np.delete(x_adtest_z1[i], -1, axis=1)
            sis2 = np.delete(sis1, -1, axis=1)
            x_adtest_z1_1.append(sis2)
            # print(sis2.shape)
        x_adtest_z1_1 = np.array(x_adtest_z1_1)
        print('x_adtest_z1_1.shape', x_adtest_z1_1.shape)
        np.save('{}'.format(address)+'/x_adtesty2', x_adtest_z1_1)



