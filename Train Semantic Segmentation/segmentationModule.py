
def segmentationModule1(X_train, x_val, y_train, y_val, BACKBONE1 = 'resnet34', epochs = 10, LR = 0.0001,batch_size = 64, optimizer = 'Adam', net_name = 'UNet' ):
    from datetime import datetime
    import tensorflow as tf
    # BACKBONE1 = 'resnet34'


    n_classes = 1;

    string_save = 'epochs_'+str(int(epochs))+'_LR_'+str(int(LR*1000000))+'_backbone_'+BACKBONE1 + 'batch'+str(int(batch_size))+optimizer
    # linkFileName = 'linknet_' + string_save
    NETFileName = net_name  + string_save
    activation = 'sigmoid'

    if(optimizer=='Adam'):
        optim = tf.keras.optimizers.Adam(LR)
    elif(optimizer=='RMSProp'):
        optim = tf.keras.optimizers.RMSprop(LR)
    elif(optimizer=='SGD'):
        optim = tf.keras.optimizers.SGD(LR)
    elif(optimizer=='Adamax'):
        optim = tf.keras.optimizers.Adamax(LR)
    elif(optimizer=='ADaDelta'):
        optim = tf.keras.optimizers.Adadelta(LR)
    elif(optimizer=='Nadam'):
        optim = tf.keras.optimizers.Nadam(LR)
    import segmentation_models as sm
    sm.set_framework('tf.keras')

    sm.framework()

    if(net_name=='UNet'):
        # print('UNet initiated')
        model = sm.Unet(BACKBONE1, input_shape=(128, 128, 3), encoder_weights='imagenet', classes=n_classes, activation=activation)
    elif(net_name=='FPN'):
        print('FPN initiated')
        model = sm.FPN(BACKBONE1, input_shape=(128, 128, 3), encoder_weights='imagenet', classes=n_classes, activation=activation)
    elif(net_name=='PSP'):
        print('PSP initiated')
        model = sm.PSPNet(BACKBONE1, input_shape=(144, 144, 3), encoder_weights='imagenet', classes=n_classes, activation=activation)
    elif(net_name=='Link'):
        print('Link initiated')
        model = sm.Linknet(BACKBONE1, input_shape=(128, 128, 3), encoder_weights='imagenet', classes=n_classes, activation=activation)


    import glob
    import cv2
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    # import keras
    import pandas as pd
    from datetime import datetime

    preprocess_input = sm.get_preprocessing(BACKBONE1)
    # Resizing images, if needed
    SIZE_X = 128
    SIZE_Y = 128
    n_classes = 1  # Number of classes for segmentation





    # X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size=0.95, random_state=42)
    # X_test, ign, y_test, ign = train_test_split(train_images, train_masks, test_size=0.95, random_state=42)
    #
    #
    # print(np.unique(X_train))
    # X_train = preprocess_input(X_train)
    # print(np.unique(X_train))
    # # # X_test = preprocess_input(X_test)
    # x_val = preprocess_input(x_val)

    #
    # y_train = preprocess_input(y_train)
    # # X_test = preprocess_input(X_test)
    # y_val = preprocess_input(y_val)


    # metrics = [ tf.keras.metrics.MeanIoU(num_classes=2), sm.metrics.FScore(threshold=0.5)]
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    #
    # print(X_train.dtype)
    # print(y_train.dtype)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    model.compile(
        optim,
        loss = sm.losses.bce_jaccard_loss,
        # loss=total_loss,
        metrics=metrics,
    )
    # model.compile(
    #     optim,
    #     loss=sm.losses.bce_jaccard_loss,
    #     metrics=metrics,
    # )

    start1 = datetime.now()
    print(X_train.dtype)
    print(y_train.dtype)
    print(x_val.dtype)
    print(y_val.dtype)
    historyUNET = model.fit(
       x=X_train,
       y=y_train,
       batch_size=batch_size,
       epochs=epochs,
        verbose=1,
       validation_data=(x_val, y_val),
    )

    stop1 = datetime.now()
    execution_time_unet = stop1 - start1
    print("Unet execution time is: ", execution_time_unet)
    # convert the history.history dict to a pandas DataFrame:
    hist1_df = pd.DataFrame(historyUNET.history)
    # hist1_df.append('something', 88)
    hist1_csv_file = 'Files/'+NETFileName + '.csv'

    hist1_df['exeTime'] = execution_time_unet
    with open(hist1_csv_file, mode='w') as f:
        hist1_df.to_csv(f)
    print(hist1_df)



    loss = historyUNET.history['loss']
    val_loss = historyUNET.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plot1 = plt.figure()
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss for '+net_name+" with "+BACKBONE1 +' as backbone')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=False)
    plot1.savefig('Figures/' + NETFileName +' loss' +'.png')
    # plot1.savefig('Figures/' + NETFileName+' loss'+'.jpg')
    plot1.savefig('Figures/' + NETFileName+' loss'+'.eps')
    acc = historyUNET.history['iou_score']
    val_acc = historyUNET.history['val_iou_score']
    plt.close()

    plot1 = plt.figure()
    plt.plot(epochs, acc, 'y', label='Training IOU')
    plt.plot(epochs, val_acc, 'r', label='Validation IOU')
    plt.title('Training and validation IOU for '+net_name+" with "+BACKBONE1 +' as backbone')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.show(block=False)
    plot1.savefig('Figures/' + NETFileName+' IOU'+'.png')
    # plot1.savefig('Figures/' + NETFileName+' IOU'+'.jpg')
    plot1.savefig('Figures/' + NETFileName+' IOU'+'.eps')

    plt.close()
    model.save('Files/'+NETFileName+'.h5')

    return model, historyUNET





def getTestTrain():
    import numpy as np
    import cv2
    import glob
    import os
    train_masks = []
    i = 0
    for directory_path in glob.glob("128 sized/masks/"):
        for mask_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            mask = cv2.imread(mask_path, 0)

            # cv2.imwrite('128 sized\\testing\masks\\'+str(i+10)+'.jpg', mask)
            i = i+1
            # uni_val  = np.unique(mask)
            # print(uni_val)
            mask = mask > 100
            mask = mask.astype('float32')
            train_masks.append(mask)
            # train_labels.append(label)
    # Convert list to array for machine learning processing
    train_masks = np.array(train_masks)
    # Capture training image info as a list
    train_images = []

    for directory_path in glob.glob("128 sized/images/"):
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            # print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = img.astype('float32')/255
            # img = cv2.resize(img, (SIZE_Y, SIZE_X))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)

    train_images = np.array(train_images)

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    train_images, train_masks = shuffle(train_images, train_masks)
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size=0.25, random_state=42)
    print('splittinh')
    X_test, x_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.666666667, random_state=42)
    print(y_test.shape, ' : is shape of validation')
    print(y_test[0].shape, ' : is shape of validation')
    for i in range(X_test.shape[0]):
        # print(np.max(X_test[i,:,:,:]))
        mask_temp = y_test[i]
        # print(mask_temp.shape)
        # # mask_temp = np.squeeze(y_val[i,:,:], axis = 0)
        # cv2.imshow('win1',mask_temp)
        # cv2.waitKey(-1)
        cv2.imwrite('128 sized/testing/images/'+str(i+10)+'.jpg', X_test[i]*255)
        cv2.imwrite('128 sized/testing/masks/'+str(i+10)+'.jpg', mask_temp*255)

    for i in range(X_train.shape[0]):
        # print(np.max(X_test[i,:,:,:]))
        mask_temp = y_train[i]
        # print(mask_temp.shape)
        # # mask_temp = np.squeeze(y_val[i,:,:], axis = 0)
        # cv2.imshow('win1',mask_temp)
        # cv2.waitKey(-1)
        cv2.imwrite('128 sized/training/images/'+str(i+10)+'.jpg', X_train[i]*255)
        cv2.imwrite('128 sized/training/masks/'+str(i+10)+'.jpg', mask_temp*255)



    for i in range(x_val.shape[0]):
        # print(np.max(X_test[i,:,:,:]))
        mask_temp = y_val[i]
        # print(mask_temp.shape)
        # # mask_temp = np.squeeze(y_val[i,:,:], axis = 0)
        # cv2.imshow('win1',mask_temp)
        # cv2.waitKey(-1)
        cv2.imwrite('128 sized/validation/images/'+str(i+10)+'.jpg', x_val[i]*255)
        cv2.imwrite('128 sized/validation/masks/'+str(i+10)+'.jpg', mask_temp*255)



    return  X_train, x_val, y_train, y_val



def getTestTrainFromDirectory():
    import numpy as np
    import cv2
    import glob
    import os
    y_train = []
    i =0
    for directory_path in glob.glob("128 sized/training/masks/"):
        for mask_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            mask = cv2.imread(mask_path, 0)

            mask = mask > 100
            mask = mask.astype('float32')
            if(i<=10):
                cv2.imshow('training', mask*255)
                cv2.waitKey(500)
                i = i+1
            y_train.append(mask)
    y_train = np.array(y_train)

    X_train = []
    i = 0
    for directory_path in glob.glob("128 sized/training/images/"):
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            # print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if(i<=10):
                cv2.imshow('traiing', img)
                cv2.waitKey(500)
                i = i+1
            img = img.astype('float32')/255
            # img = cv2.resize(img, (SIZE_Y, SIZE_X))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            X_train.append(img)

    X_train = np.array(X_train)


    y_val = []
    i =0
    for directory_path in glob.glob("128 sized/validation/masks/"):
        for mask_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            mask = cv2.imread(mask_path, 0)

            mask = mask > 100
            mask = mask.astype('float32')
            if(i<=10):
                cv2.imshow('validation', mask*255)
                cv2.waitKey(500)
                i = i+1
            y_val.append(mask)
    y_val = np.array(y_val)

    x_val = []
    i = 0
    for directory_path in glob.glob("128 sized/validation/images/"):
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            # print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if(i<=10):
                cv2.imshow('validation', img)
                cv2.waitKey(500)
                i = i+1
            img = img.astype('float32')/255
            # img = cv2.resize(img, (SIZE_Y, SIZE_X))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            x_val.append(img)

    x_val = np.array(x_val)


    return  X_train, x_val, y_train, y_val




 # Testing_IOU
