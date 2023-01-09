
import os
import segmentationModule as mySM
from os.path import exists
import time
import random
import numpy as np
from UScratchASAN import getTestTrainFromDirectory, compute_iou
import tqdm
import cv2
print('hi')


from openpyxl import Workbook
import openpyxl
import pandas as pd
workbook = Workbook()


worksheet = workbook.active

X_test, Y_test = getTestTrainFromDirectory('D:\SIBASAN\\testing\\',True)
print(Y_test.dtype)
print(X_test.dtype)

print(np.unique(Y_test))
print(np.unique(X_test))
print( ' is test shape')
# time.sleep(100)
X_train, Y_train = getTestTrainFromDirectory('D:\SIBASAN\\training\\',True)
from sklearn.utils import shuffle
X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

print(Y_train.dtype)
print(X_train.dtype)
print( ' is train shape')
X_val, y_val = getTestTrainFromDirectory('D:\SIBASAN\\validation\\',True)
print(y_val.dtype)
print(' is val shape')



epochs = 100
# LR = 0.0001
# LRS = [0.00001, 0.0001]
LRS = [0.0001]
# batchS = [128, 92, 64, 32, 16]
batchS = [32, 128, 16, 64]
# batchS = [192, 128, 64, 32, 16]
optims = ['Adam', 'RMSProp']
# optims = ['Adam', 'RMSProp', 'SGD', 'ADaDelta']
# optims = ['Adam', 'RMSProp', 'SGD', 'Adamax', 'ADaDelta', 'Nadam']
nets = [ 'FPN', 'UNet',  'Link']
# nets = [ 'UNet', 'FPN', 'Link','PSP']
backs = ['vgg16', 'resnet18', 'vgg19','resnet50', 'resnet152', 'inceptionv3', 'mobilenetv2']
# backs = ['vgg16', 'vgg19','resnet18',  'resnet152','resnet50','inceptionv3'] # , 'mobilenetv2'
# backs = ['vgg16', 'vgg19','resnet18',  'resnet50',  'resnet152','densenet121', 'densenet169', 'densenet201','inceptionv3' ,'inceptionresnetv2','efficientnetb0',  'efficientnetb1']
count = 0
for LR in LRS:
    for batch_size in batchS:
        for optimizer in optims:
                for BACKBONE1 in backs:
                    for net_name in nets:
                        string_save = 'epochs_' + str(int(epochs)) + '_LR_' + str(int(LR * 1000000)) + '_backbone_' + BACKBONE1 + 'batch' + str(int(batch_size)) + optimizer
                        # linkFileName = 'linknet_' + string_save
                        NETFileName = net_name + string_save
                        file_exists = exists('Files\\' + NETFileName + '.h5')
                        if(file_exists):
                            print('file already exist, skipping')
                            print('Files\\' + NETFileName + '.h5')
                            continue
                        print(NETFileName + "................ Running")
                        # try:
                        count = count + 1



                        print(X_train.dtype)
                        print(X_val.dtype)
                        print(Y_train.dtype)
                        print(y_val.dtype)
                        # time.sleep(100)
                        model, modelHistory = mySM.segmentationModule1(X_train, X_val, Y_train, y_val, BACKBONE1=BACKBONE1, epochs=epochs, LR=LR, batch_size=batch_size, optimizer=optimizer, net_name=net_name)
                        idx = random.randint(0, len(X_train))

                        # preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
                        # preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
                        preds_test = model.predict(X_test, verbose=1)

                        # preds_train_t = (preds_train > 0.5).astype(np.uint8)
                        # preds_val_t = (preds_val > 0.5).astype(np.uint8)
                        preds_test_t = (preds_test > 0.5).astype(np.uint8)

                        mean_iou = 0
                        sum_iou = 0
                        countI = 0
                        os.makedirs('D:\SIBASAN\\test output\\'+NETFileName)
                        for i in range(len(preds_test_t)):
                            img = X_test[i]
                            mask = 1 * Y_test[i]
                            pred = preds_test_t[i]

                            print(compute_iou(mask, pred), '   ...... is IOU')
                            sum_iou = sum_iou + compute_iou(mask, pred)
                            pred = np.squeeze(pred)
                            pred = pred*255
                            mask = np.squeeze(mask) * 255
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)*255
                            cv2.imwrite('D:\SIBASAN\\test output\\' + NETFileName + '\\' + str(i) + ' 1.png', img)
                            cv2.imwrite('D:\SIBASAN\\test output\\' + NETFileName + '\\' + str(i) + ' 2.png', mask)
                            cv2.imwrite('D:\SIBASAN\\test output\\' + NETFileName + '\\' + str(i) + ' 3.png', pred)
                            countI = countI + 1
                            # print('done............')
                        mean_iou = sum_iou/len(preds_test_t)
                        print(mean_iou,' is mean iou')
                        print(countI, ' is count iou')
                        print(sum_iou,' is sum iou')
                        print(len(preds_test_t), ' is prediction length')

                        worksheet.append([net_name, str(int(epochs)),'LR: ' + str(int(LR * 1000000)), BACKBONE1,  str(int(batch_size)), optimizer, str(modelHistory.history['iou_score'][-1]*100),str(modelHistory.history['val_iou_score'][-1] * 100), str(modelHistory.history['loss'][-1]*100), str(modelHistory.history['val_loss'][-1] * 100), str(mean_iou*100)])
                        workbook.save('Results.xlsx')
                        # worksheet.write('A' + str(count), net_name)
                        # worksheet.write('B' + str(count), str(int(epochs)))
                        # worksheet.write('C' + str(count), 'LR: ' + str(int(LR * 1000000)))
                        # worksheet.write('D' + str(count), BACKBONE1)
                        # worksheet.write('E' + str(count), str(int(batch_size)))
                        # worksheet.write('F' + str(count), optimizer)
                        # worksheet.write('G' + str(count), str(modelHistory.history['iou_score'][-1]*100))
                        # worksheet.write('H' + str(count), str(modelHistory.history['val_iou_score'][-1] * 100))
                        # worksheet.write('I' + str(count), str(modelHistory.history['loss'][-1]*100))
                        # worksheet.write('J' + str(count), str(modelHistory.history['val_loss'][-1] * 100))
                        # worksheet.write('K' + str(count), str(mean_iou*100))
                        # workbook.close()
                        print('workbook closed')
                        time.sleep(120)
                        # except:
                        #     print(NETFileName)
                        # break
                    # break
                # break
        # break
    # break

                    # break

# workbook.close()