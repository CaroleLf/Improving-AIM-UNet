import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size, mask_interpolation=cv2.INTER_NEAREST),
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
                  mask_interpolation=cv2.INTER_NEAREST, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_eval_transform(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size, mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ToTensorV2(),
    ])
