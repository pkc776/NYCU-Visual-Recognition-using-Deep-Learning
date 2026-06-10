import os
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CellDataset(Dataset):
    def __init__(self, root_dir, split="train", transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., path to 'train' or 'test').
            split (string): "train", "val", or "test".
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        # In train/val split, root_dir is something like "data/train"
        # Inside "data/train", there are subdirectories for each image.
        if self.split in ["train", "val"]:
            self.image_dirs = sorted([
                d for d in os.listdir(self.root_dir) 
                if os.path.isdir(os.path.join(self.root_dir, d))
            ])
        else: # test split
            # In test split, root_dir is something like "data/test"
            # Inside "data/test", there are directly .tif files
            self.image_files = sorted([
                f for f in os.listdir(self.root_dir)
                if f.endswith('.tif')
            ])

    def __len__(self):
        if self.split in ["train", "val"]:
            return len(self.image_dirs)
        else:
            return len(self.image_files)

    def __getitem__(self, idx):
        if self.split in ["train", "val"]:
            img_dir_name = self.image_dirs[idx]
            img_dir = os.path.join(self.root_dir, img_dir_name)
            
            # Read image
            img_path = os.path.join(img_dir, "image.tif")
            image = tifffile.imread(img_path)
            
            # If image has shape (C, H, W), we might need to transpose to (H, W, C) for albumentations
            # Tifffile usually reads images as (H, W, C) or (H, W), but let's ensure it's HWC or HW
            if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]: # likely CHW
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure it's RGB
            if len(image.shape) == 2:
                image = np.stack([image]*3, axis=-1)
            elif image.shape[-1] == 4: # RGBA
                image = image[..., :3]

            boxes = []
            labels = []
            masks = []
            
            # Read masks
            for class_id in range(1, 5): # classes 1 to 4
                mask_path = os.path.join(img_dir, f"class{class_id}.tif")
                if not os.path.exists(mask_path):
                    continue
                
                class_mask = tifffile.imread(mask_path)
                
                # each unique value represents an instance
                instance_ids = np.unique(class_mask)
                # remove background (0)
                instance_ids = instance_ids[instance_ids != 0]
                
                for inst_id in instance_ids:
                    inst_mask = (class_mask == inst_id).astype(np.uint8)
                    
                    # Compute bounding box
                    pos = np.where(inst_mask)
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    
                    # If the mask is empty or 1-pixel wide/high, we might need to adjust or skip
                    if xmax <= xmin or ymax <= ymin:
                        # try to add 1 to max to ensure positive width/height
                        if xmax == xmin:
                            xmax += 1
                        if ymax == ymin:
                            ymax += 1
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)
                    masks.append(inst_mask)

            # Convert to numpy arrays
            if len(boxes) > 0:
                boxes = np.array(boxes, dtype=np.float32)
                labels = np.array(labels, dtype=np.int64)
                masks = np.stack(masks, axis=0) # (N, H, W)
            else:
                # Handle empty images
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
                masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # Apply transforms
            if self.transforms is not None:
                masks_list = [m for m in masks] if len(masks) > 0 else []
                augmented = self.transforms(image=image, masks=masks_list)
                image = augmented['image'].float() / 255.0
                aug_masks = augmented['masks']
                
                valid_boxes = []
                valid_labels = []
                valid_masks = []
                for i, m in enumerate(aug_masks):
                    # m is a PyTorch tensor here because of ToTensorV2
                    if m.any():
                        pos = torch.where(m)
                        xmin = torch.min(pos[1]).item()
                        xmax = torch.max(pos[1]).item()
                        ymin = torch.min(pos[0]).item()
                        ymax = torch.max(pos[0]).item()
                        if xmax > xmin and ymax > ymin:
                            valid_boxes.append([xmin, ymin, xmax, ymax])
                            valid_labels.append(labels[i])
                            valid_masks.append(m.numpy())
                
                if len(valid_boxes) > 0:
                    boxes = np.array(valid_boxes, dtype=np.float32)
                    labels = np.array(valid_labels, dtype=np.int64)
                    masks = np.stack(valid_masks, axis=0)
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    labels = np.zeros((0,), dtype=np.int64)
                    masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)
            else:
                # Basic transform to PyTorch tensors
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            
            image_id = torch.tensor([idx])
            if len(boxes) > 0:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            else:
                area = torch.zeros((0,), dtype=torch.float32)
                
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            return image, target
        
        else: # test split
            img_file = self.image_files[idx]
            img_path = os.path.join(self.root_dir, img_file)
            image = tifffile.imread(img_path)
            
            if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
                
            if len(image.shape) == 2:
                image = np.stack([image]*3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
                
            if self.transforms is not None:
                augmented = self.transforms(image=image)
                image = augmented['image'].float() / 255.0
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                
            return image, img_file

MAX_SIZE = 800  # Limit max image size to avoid OOM

def get_train_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=MAX_SIZE, p=1.0),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,
                      pad_width_divisor=32, border_mode=0, value=0, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5, border_mode=0),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ], p=0.3),
        ToTensorV2(p=1.0)
    ])

def get_val_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=MAX_SIZE, p=1.0),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,
                      pad_width_divisor=32, border_mode=0, value=0, p=1.0),
        ToTensorV2(p=1.0)
    ])

def get_test_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=MAX_SIZE, p=1.0),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32,
                      pad_width_divisor=32, border_mode=0, value=0, p=1.0),
        ToTensorV2(p=1.0)
    ])
