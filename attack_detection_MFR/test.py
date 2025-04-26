from __future__ import print_function
import pandas as pd
import numpy as np
import xlrd
# file='imag_val_lable.xlsx'
# wb=xlrd.open_workbook(filename=file)
# ws=wb.sheet_by_name('Sheet1')
# dataset=[]
# for r in range(ws.nrows):
#     clo=[]
#     for c in range(ws.ncols):
#         clo.append(ws.cell(r,c).value)
#     dataset.append(clo)
# dataset=np.array(dataset)
# print(dataset.shape)
# print(dataset)
# np.save('imag_val_lable',dataset)

x_test=np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidence4901.npy')
y_test=np.load('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_adconfidence4901.npy')
x_test2000=x_test[0:500]
y_test2000=y_test[0:500]
print(x_test2000.shape)
print(np.argmax(x_test[40]))
print(y_test2000.shape)
print(np.argmax(y_test[40]))


np.save('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_confidence00',x_test2000)
np.save('/home/Bear/attack_detection/fgsm_resnet_imagenet/y_adconfidence00',y_test2000)


