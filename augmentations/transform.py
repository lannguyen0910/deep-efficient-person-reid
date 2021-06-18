# import cv2
# import numpy as np
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
# from .random_erasing import RandomErasing

# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]


# class Denormalize(object):
#     """
#     Denormalize image and boxes for visualization
#     """

#     def __init__(self, mean=MEAN, std=STD, **kwargs):
#         self.mean = mean
#         self.std = std

#     def __call__(self, img, box=None, label=None, mask=None, **kwargs):
#         """
#         :param img: (tensor) image to be denormalized
#         :param box: (list of tensor) bounding boxes to be denormalized, by multiplying them with image's width and heights. Format: (x,y,width,height)
#         """
#         mean = np.array(self.mean)
#         std = np.array(self.std)
#         img_show = img.numpy().squeeze().transpose((1, 2, 0))
#         img_show = (img_show * std+mean)
#         img_show = np.clip(img_show, 0, 1)
#         return img_show


# def get_resize_augmentation(image_size, keep_ratio=False, box_transforms=False):

#     if not keep_ratio:
#         return A.Compose([
#         ])
#     else:
#         return A.Compose([
#             A.LongestMaxSize(max_size=max(image_size)),
#             A.PadIfNeeded(
#                 min_height=image_size[1], min_width=image_size[0], p=1.0, border_mode=cv2.BORDER_CONSTANT),
#         ])


# PROB: 0.5  # random horizontal flip
# RE_PROB: 0.5  # random erasing


# def get_augmentation(config, _type='train'):
#     train_transforms = A.Compose([
#         A.Resize(
#             height=config.image_size[1],
#             width=config.image_size[0]),
#         A.OneOf([
#             A.MotionBlur(p=.2),
#             A.GaussianBlur(),
#             A.MedianBlur(blur_limit=3, p=0.3),
#             A.Blur(blur_limit=3, p=0.1),
#         ], p=0.3),
#         A.ShiftScaleRotate(shift_limit=0.0625,
#                            scale_limit=0.2, rotate_limit=20, p=0.3),
#         # A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
#         A.OneOf([
#             A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
#                                  val_shift_limit=0.2, p=0.9),
#             A.RandomBrightnessContrast(brightness_limit=0.1,
#                                        contrast_limit=0.1,
#                                        p=0.3),
#         ], p=0.5),

#         A.HorizontalFlip(p=0.3),
#         A.VerticalFlip(p=0.3),
#         A.RandomRotate90(p=0.3),
#         A.Cutout(num_holes=8, max_h_size=64,
#                  max_w_size=64, fill_value=0, p=0.5),
#         A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1.0),
#         ToTensorV2(p=1.0),
#         RandomErasing()])

#     val_transforms = A.Compose([
#         A.Resize(
#             height=config.image_size[1],
#             width=config.image_size[0]),
#         A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1.0),
#         ToTensorV2(p=1.0)])

#     test_transforms = A.Compose([
#         A.RandomResizedCrop(height=config.image_size[1],
#                             width=config.image_size[0]),
#         A.Transpose(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.HueSaturationValue(hue_shift_limit=0.2,
#                              sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#         A.RandomBrightnessContrast(
#             brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
#         A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0, p=1.0),
#         ToTensorV2(p=1.0),
#     ], p=1.)

#     if _type == 'train':
#         return train_transforms
#     elif _type == 'val':
#         return val_transforms
#     elif _type == 'test':
#         return test_transforms
import torchvision.transforms as T

from .random_erasing import RandomErasing


def build_transforms(config, is_train=True):
    normalize_transform = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        transform = T.Compose([
            T.Resize(config.image_size),
            T.RandomHorizontalFlip(p=config.prob),
            T.Pad(config.padding),
            T.RandomCrop(config.image_size),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=config.re_prob,
                          mean=[0.485, 0.456, 0.406])
        ])
    else:
        transform = T.Compose([
            T.Resize(config.image_size),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
