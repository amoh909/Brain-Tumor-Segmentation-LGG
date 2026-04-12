import albumentations as A


def get_train_transforms():
    return A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Affine(
        #     translate_percent=0.06,
        #     scale=(0.9, 1.1),
        #     rotate=(-20, 20),
        #     p=0.5
        # ),
        # A.RandomBrightnessContrast(
        #     brightness_limit=0.2,
        #     contrast_limit=0.2,
        #     p=0.3
        # ), for the sake of E2
    ])


def get_val_transforms(): 
    return A.Compose([])