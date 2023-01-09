import tensorflow as tf
import os
import random
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# import numpy as np

def compute_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ground_truth = np.array(ground_truth)
    pred = np.array(pred)
    un_both = np.sum((ground_truth+pred)>0)
    in_both = np.sum(ground_truth*pred)
    iou = in_both/un_both

    return iou




def getIntensityOnly(path, toFloat32=False):

    import os
    from skimage.io import imread, imshow
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
    images_name = os.listdir(path)
    num_images = len(images_name)
    X = np.zeros((num_images, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y = np.zeros((num_images, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(images_name), total=num_images):
        img = imread(path  + id_)[:, :, :IMG_CHANNELS]
        # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        if(toFloat32):
            img = img.astype('float32')/255




        X[n] = img  # Fill empty X_train with values from img


    return  X



def getTestTrainFromDirectory(path, toFloat32=False):

    import os
    from skimage.io import imread, imshow
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
    images_name = os.listdir(path+'images')
    num_images = len(images_name)
    X = np.zeros((num_images, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    Y = np.zeros((num_images, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(images_name), total=num_images):
        img = imread(path + '\\images\\' + id_)[:, :, :IMG_CHANNELS]
        # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        mask_1 = imread(path + '\\masks\\' + id_[:-3] + 'png')
        if(toFloat32):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
            mask_1 = mask_1.astype('float32') / 255
            img = img.astype('float32')/255
        else:
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


        if(mask_1.ndim==3):
            mask_ = mask_1[:,:,0]

        else:
            mask_ = mask_1
        X[n] = img  # Fill empty X_train with values from img
        # print(np.unique(mask_1))
        # mask2 = imread(path + '\\masks\\' + id_[:-3]+'jpg')
        # print(np.unique(mask_))

        # print(np.unique(mask2))
        mask_ = np.expand_dims(mask_, axis=-1)
        mask = np.maximum(mask, mask_)
        Y[n] = mask

    return  X, Y

if __name__ == '__main__':
    seed = 42
    np.random.seed = seed

    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    # TRAIN_PATH = 'D:\SIBASAN\\testing\images\'
    # TEST_PATH = 'D:\Sreeni\\stage1_test\\'
    # VAL_PATH = 'D:\Sreeni\\stage1_test\\'
    #
    #
    # train_ids = next(os.walk(TRAIN_PATH))[1]
    # test_ids = next(os.walk(TEST_PATH))[1]
    #
    #
    # X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    # Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    #
    # print('Resizing training images and masks')
    # for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    #     path = TRAIN_PATH + id_
    #     img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    #     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #     X_train[n] = img  # Fill empty X_train with values from img
    #     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    #     for mask_file in next(os.walk(path + '/masks/'))[2]:
    #         mask_ = imread(path + '/masks/' + mask_file)
    #         mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
    #                                       preserve_range=True), axis=-1)
    #         mask = np.maximum(mask, mask_)
    #
    #     Y_train[n] = mask
    # # from sklearn.utils import shuffle
    # # X_train, Y_train = shuffle(X_train, Y_train)
    # # X_train, X_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.17, random_state=42)
    # # test images
    # X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    # sizes_test = []
    # print('Resizing test images')
    # for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    #     path = TEST_PATH + id_
    #     img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    #     sizes_test.append([img.shape[0], img.shape[1]])
    #     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    #     X_test[n] = img
    #
    # print('Done!')

    # image_x = random.randint(0, len(train_ids))
    # imshow(X_train[image_x])
    # plt.show()
    # imshow(np.squeeze(Y_train[image_x]))
    # plt.show()

    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Build the model
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.MeanIoU(num_classes=2)])
    model.summary()

    ################################
    #Modelcheckpoint
    checkpointer = tf.keras.callbacks.ModelCheckpoint('SIBASAN.h5', verbose=1, save_best_only=True)

    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs')]
    epochs = 200
    X_train, Y_train = getTestTrainFromDirectory('D:\SIBASAN\\training\\')
    from sklearn.utils import shuffle
    X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
    X_val, y_val = getTestTrainFromDirectory('D:\SIBASAN\\validation\\')
    X_test, Y_test = getTestTrainFromDirectory('D:\SIBASAN\\testing\\')

    print('printing types')
    print(X_train.dtype)
    print(Y_train.dtype)
    print(X_val.dtype)
    print(y_val.dtype)


    # results = model.fit(X_train, Y_train, batch_size=16, epochs=epochs, callbacks=callbacks, validation_data=(X_val, y_val))
    results = model.fit(X_train, Y_train, batch_size=64, epochs=epochs, validation_data=(X_val, y_val))


    acc = results.history['mean_io_u']
    epochsNum = range(1, len(acc) + 1)
    plt.close()

    val_acc = results.history['val_mean_io_u']
    plot1 = plt.figure()
    plt.plot(epochsNum, acc, 'y', label='Training IOU')
    plt.plot(epochsNum, val_acc, 'r', label='Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.show()
    ####################################
    idx = random.randint(0, len(X_train))

    preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
    preds_test = model.predict(X_test, verbose=1)

    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    import cv2
    for i in range(len(preds_test_t)):
        img = X_test[i]
        mask = 1*Y_test[i]
        pred = preds_test_t[i]*255
        pred = np.squeeze(pred)
        mask = np.squeeze(mask)*255
        cv2.imwrite('D:\SIBASAN\\test output\\'+str(i)+' 1.jpg',img)
        cv2.imwrite('D:\SIBASAN\\test output\\'+str(i)+' 2.jpg',mask)
        cv2.imwrite('D:\SIBASAN\\test output\\'+str(i)+' 3.jpg',pred)


    # Perform a sanity check on some random training samples
    # ix = random.randint(0, len(preds_train_t))


    # imshow(X_train[ix])
    # plt.show()
    # print(Y_train.shape)
    # imshow(np.squeeze(Y_train[ix]))
    # plt.show()
    # imshow(np.squeeze(preds_train_t[ix]))
    # plt.show()
    #
    # # Perform a sanity check on some random validation samples
    # ix = random.randint(0, len(preds_val_t))
    # imshow(X_train[int(X_train.shape[0] * 0.9):][ix])
    # plt.show()
    # imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9):][ix]))
    # plt.show()
    # imshow(np.squeeze(preds_val_t[ix]))
    # plt.show()

