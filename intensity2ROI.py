import copy

import tensorflow as tf
from tensorflow import keras
from UScratchASAN import getIntensityOnly
import numpy as np
import cv2
import matplotlib.pyplot as plt


# epochs = 100
# LR = 0.00001
# BACKBONE1 = 'vgg16'
# batch_size = 64
# optimizer = 'Adam'
# net_name = 'UNET'
#
# string_save = 'epochs_' + str(int(epochs)) + '_LR_' + str(int(LR * 1000000)) + '_backbone_' + BACKBONE1 + 'batch' + str(int(batch_size)) + optimizer
# linkFileName = 'linknet_' + string_save
# NETFileName = net_name + string_save
NETFileName = '1'
model = keras.models.load_model('Saved Seg Models/'+NETFileName+'.h5', compile=False)
FolderName = 'Samples intensity images'
X_test = getIntensityOnly('Dataset/'+ FolderName+'/',True)
# plt.imshow(a)
# plt.show()
preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

mean_iou = 0
sum_iou = 0
countI = 0
for i in range(len(preds_test_t)):
    img = X_test[i]
    pred = preds_test_t[i]
    pred = np.squeeze(pred)
    image_lesion = copy.copy(img)
    image_lesion[:,:,0] = image_lesion[:,:,0]*pred
    image_lesion[:,:,1] = image_lesion[:,:,1]*pred
    image_lesion[:,:,2] = image_lesion[:,:,2]*pred
    pred = pred*255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) * 255
    img = img.astype(np.uint8)
    image_lesion = cv2.cvtColor(image_lesion, cv2.COLOR_RGB2BGR) * 255
    image_lesion = image_lesion.astype(np.uint8)

    cv2.imwrite( FolderName + 'ROI' + '/' + str(i) + ' Mask.png', pred)
    cv2.imwrite(FolderName + '/' + str(i) + ' ROI.png', image_lesion)
    # print('done............')