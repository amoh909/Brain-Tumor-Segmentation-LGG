import albumentations as A


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2), ## Stretchy Distortions
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(p=0.2),
    ])


def get_val_transforms(): ## We don't do data augmentations on the validation/test sets, this function is added to normalize later
    return A.Compose([])