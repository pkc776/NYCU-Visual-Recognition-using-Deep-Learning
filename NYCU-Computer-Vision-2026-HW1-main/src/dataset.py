import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def get_transforms(train=True, aug_type="randaug"):
    if train:
        base_transforms = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]

        if aug_type == "colorjitter":
            base_transforms.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )
            )
        elif aug_type == "randaug":
            base_transforms.append(transforms.RandAugment())
        elif aug_type == "autoaug":
            base_transforms.append(
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
            )

        base_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transforms.Compose(base_transforms)
    else:
        # Standard transformations for validation/testing
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        filename = os.path.basename(img_path)
        return image, filename


def get_dataloaders(
    data_dir, batch_size=32, num_workers=4, aug_type="randaug"
):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(
        train_dir, transform=get_transforms(train=True, aug_type=aug_type)
    )
    val_dataset = datasets.ImageFolder(
        val_dir, transform=get_transforms(train=False)
    )
    test_dataset = TestDataset(test_dir, transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
