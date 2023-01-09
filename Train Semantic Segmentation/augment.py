# # import the modules
# import os
# from os import listdir

# get the path/directory
import os
import numpy as np
import cv2
from glob import glob
# from tqdm import tqdm
# import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, MotionBlur, RandomBrightness, RandomGamma, CLAHE, ColorJitter, CoarseDropout, RandomCrop, Downscale, RandomContrast, RandomBrightnessContrast



def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    """ X = Images and Y = masks """

    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)

def augment_data():
    H = 104
    W = 104

    # folder_dir = "F:\Dataset_for_Augmentation\Intensity"
    # folder_dir2 = "F:\Dataset_for_Augmentation\GT"

    folder_dir = "D:\\Augmentation_dataset\\Intensity"

    folder_dir2 = "D:\\Augmentation_dataset\\GT"


    for images in os.listdir(folder_dir):

        # check if the image ends with png
        if (images.endswith(".jpg")):
            x = cv2.imread("D:\\Augmentation_dataset\\Intensity\\" + images)
            y = cv2.imread("D:\\Augmentation_dataset\\GT\\" + images)

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            # aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            # augmented = aug(image=x, mask=y)
            # x3 = augmented['image']
            # y3 = augmented['mask']

            aug = GridDistortion(p=0.5, distort_limit=(-0.03, 0.02))
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = OpticalDistortion(p=1, distort_limit=1.5, shift_limit=0.05)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            aug = RandomBrightness(limit=0.2, always_apply=False, p=0.5)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            aug = MotionBlur(blur_limit=5, p=0.5)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']

            aug = RandomGamma(gamma_limit=(30, 35), eps=None, always_apply=False, p=0.5)
            augmented = aug(image=x, mask=y)
            x8 = augmented['image']
            y8 = augmented['mask']

            aug = CLAHE(clip_limit=(1.16, 1.25), tile_grid_size=(16, 16), always_apply=False, p=0.5)
            augmented = aug(image=x, mask=y)
            x9 = augmented['image']
            y9 = augmented['mask']

            aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5)
            augmented = aug(image=x, mask=y)
            x10 = augmented['image']
            y10 = augmented['mask']

            aug = Downscale(scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.5)
            augmented = aug(image=x, mask=y)
            x11 = augmented['image']
            y11 = augmented['mask']




            # Save augmented Images into Directory


            # Original Image
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x.jpg", x)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x.jpg", y)

            # 1. Horizontal Flip
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x1.jpg", x1)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x1.jpg", y1)

            # 2. Vertical Flip
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x2.jpg", x2)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x2.jpg", y2)

            # 3. Elastic Transform
            # cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x3.jpg", x3)
            # cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x3.jpg", y3)

            # 4. Grid Distortion
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x4.jpg", x4)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x4.jpg", y4)

            # 5. Optical Distortion
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x5.jpg", x5)
            cv2.imwrite("D:\\augmentation\gt\\" + images[:-4] + "x5.jpg", y5)

            # 6. Random Brightness
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x6.jpg", x6)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x6.jpg", y6)

            # 7. Motion Blur
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x7.jpg", x7)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x7.jpg", y7)

            # 8. Random Gamma
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x8.jpg", x8)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x8.jpg", y8)

            # 9. Clahe
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x9.jpg", x9)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x9.jpg", y9)

            # 10 . Color Jitter
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x10.jpg", x10)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x10.jpg", y10)

            # 11 . Downscale
            cv2.imwrite("D:\\augmentation\\intensity\\" + images[:-4] + "x11.jpg", x11)
            cv2.imwrite("D:\\augmentation\\gt\\" + images[:-4] + "x11.jpg", y11)


        #     X = [x, x1, x2, x3, x4, x5]
        #     Y = [y, y1, y2, y3, y4, y5]
        #
        # else:
        #     X = [x]
        #     Y = [y]
        #
        # index = 0
        # for i, m in zip(X, Y):
        #     i = cv2.resize(i, (W, H))
        #     m = cv2.resize(m, (W, H))
        #
        #     if len(X) == 1:
        #         tmp_image_name = f"{name}.jpg"
        #         tmp_mask_name = f"{name}.jpg"
        #     else:
        #         tmp_image_name = f"{name}_{index}.jpg"
        #         tmp_mask_name = f"{name}_{index}.jpg"
        #
        #     image_path = os.path.join(save_path, "image", tmp_image_name)
        #     mask_path = os.path.join(save_path, "mask", tmp_mask_name)
        #
        #     cv2.imwrite(image_path, i)
        #     cv2.imwrite(mask_path, m)
        #
        #     index += 1

if __name__ == "__main__":
    """ Seeding """

    augment_data()

    # print("hi")
    # y = cv2.imread("F:\Dataset_for_Augmentation\GT\Ak_0.jpg")
    # x = cv2.imread("F:\Dataset_for_Augmentation\Intensity\Ak_0.jpg")
    # # cv2.imshow("window1", x)
    # # cv2.imshow("window2", y)
    # # cv2.waitKey(-1)
    #
    # aug = HorizontalFlip(p=1.0)
    # augmented = aug(image=x, mask=y)
    # x1 = augmented["image"]
    # y1 = augmented["mask"]
    #
    # aug = VerticalFlip(p=1.0)
    # augmented = aug(image=x, mask=y)
    # x2 = augmented["image"]
    # y2 = augmented["mask"]
    #
    # aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
    # augmented = aug(image=x, mask=y)
    # x3 = augmented['image']
    # y3 = augmented['mask']
    #
    # aug = GridDistortion(p=1)
    # augmented = aug(image=x, mask=y)
    # x4 = augmented['image']
    # y4 = augmented['mask']
    #
    # aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
    # augmented = aug(image=x, mask=y)
    # x5 = augmented['image']
    # y5 = augmented['mask']
    #
    # aug = RandomBrightness (limit=0.2, always_apply=False, p=0.5)
    # augmented = aug(image=x, mask=y)
    # x6 = augmented['image']
    # y6 = augmented['mask']
    #
    # aug = MotionBlur (blur_limit=4, p=0.5)
    # augmented = aug(image=x, mask=y)
    # x7 = augmented['image']
    # y7 = augmented['mask']
    #
    # aug = RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5)
    # augmented = aug(image=x, mask=y)
    # x8 = augmented['image']
    # y8 = augmented['mask']
    #
    # cv2.imwrite("F:\\augmentation\\intensity\\x1.jpg", x1)
    # cv2.imwrite("F:\\augmentation\\gt\\y1.jpg", y1)
    #
    # cv2.imwrite("F:\\augmentation\\intensity\\x2.jpg", x2)
    # cv2.imwrite("F:\\augmentation\\gt\\y2.jpg", y2)
    #
    # cv2.imwrite("F:\\augmentation\\intensity\\x3.jpg", x3)
    # cv2.imwrite("F:\\augmentation\\gt\\y3.jpg", y3)
    #
    # cv2.imwrite("F:\\augmentation\\intensity\\x4.jpg", x4)
    # cv2.imwrite("F:\\augmentation\gt\y4.jpg", y4)
    #
    # cv2.imwrite("F:\\augmentation\\intensity\\x5.jpg", x5)
    # cv2.imwrite("F:\\augmentation\gt\y5.jpg", y5)
    #
    # cv2.imwrite("F:\\augmentation\intensity\\x6.jpg", x6)
    # cv2.imwrite("F:\\augmentation\gt\y6.jpg", y6)
    #
    # cv2.imwrite("F:\\augmentation\intensity\\x7.jpg", x7)
    # cv2.imwrite("F:\\augmentation\gt\y7.jpg", y7)
    #
    # cv2.imwrite("F:\\augmentation\intensity\\x8.jpg", x8)
    # cv2.imwrite("F:\\augmentation\gt\y8.jpg", y8)


    #
    # aug = HorizontalFlip(p=1.0)
    # augmented = aug(image=x, mask=y)
    # x1 = augmented["image"]
    # y1 = augmented["mask"]
    #
    #
    # np.random.seed(42)
    #
    # """ Load the data """
    # data_path = "/media/nikhil/ML/ml_dataset/Retina blood vessel segmentation/"
    # (train_x, train_y), (test_x, test_y) = load_data(data_path)
    #
    # print(f"Train: {len(train_x)} - {len(train_y)}")
    # print(f"Test: {len(test_x)} - {len(test_y)}")
    #
    # """ Creating directories """
    # create_dir("new_data/train/image")
    # create_dir("new_data/train/mask")
    # create_dir("new_data/test/image")
    # create_dir("new_data/test/mask")
    #
    # augment_data(train_x, train_y, "new_data/train/", augment=False)
    # augment_data(test_x, test_y, "new_data/test/", augment=False)
