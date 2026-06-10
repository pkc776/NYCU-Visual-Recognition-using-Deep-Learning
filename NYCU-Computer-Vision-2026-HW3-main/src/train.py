import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    print("Please install torchmetrics: pip install torchmetrics")

from dataset import CellDataset, get_train_transforms, get_val_transforms
from model import get_model_instance_segmentation

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, grad_accum_steps=1, scaler=None):
    model.train()
    
    total_loss = 0.0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    optimizer.zero_grad()
    
    for step, (images, targets) in enumerate(progress_bar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses / grad_accum_steps).backward()
            loss_value = losses.item()
            total_loss += loss_value
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(data_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            (losses / grad_accum_steps).backward()
            
            loss_value = losses.item()
            total_loss += loss_value
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(data_loader):
                optimizer.step()
                optimizer.zero_grad()
        
        # update progress bar
        progress_bar.set_postfix({'loss': f"{loss_value:.4f}"})
        
    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    
    metric = MeanAveragePrecision(iou_type="segm")
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Val]")
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        
        # When model is in eval mode, it returns predictions
        predictions = model(images)
        
        # Move targets to cpu for metric calculation
        cpu_targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
        cpu_preds = []
        for p in predictions:
            cpu_p = {
                'boxes':  p['boxes'].cpu(),
                'scores': p['scores'].cpu(),
                'labels': p['labels'].cpu(),
                # convert to bool mask immediately to save memory
                'masks':  (p['masks'].cpu() > 0.5).squeeze(1),
            }
            cpu_preds.append(cpu_p)
        
        # Explicitly free GPU predictions before metric update
        del predictions, images
        torch.cuda.empty_cache()
        
        metric.update(cpu_preds, cpu_targets)
        del cpu_preds, cpu_targets
        
    results = metric.compute()
    print(f"Validation AP50: {results['map_50']:.4f}")
    return results['map_50']

def main():
    # Configuration
    data_dir = "train" # Assuming data is extracted at current directory
    batch_size = 2
    grad_accum_steps = 2  # effective batch size = batch_size * grad_accum_steps = 4
    num_epochs = 100
    learning_rate = 1e-4
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 5 # 4 cell classes + 1 background
    
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    full_dataset = CellDataset(data_dir, split="train", transforms=get_train_transforms())
    
    # Split dataset (e.g., 90% train, 10% val)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Val dataset should use val transforms
    val_dataset.dataset.transforms = get_val_transforms() # Note: this applies to full dataset but it's shared, so we should actually instantiate another dataset
    
    # Proper way to handle different transforms for train and val splits
    train_dataset = CellDataset(data_dir, split="train", transforms=get_train_transforms())
    val_dataset = CellDataset(data_dir, split="train", transforms=get_val_transforms())
    
    indices = torch.randperm(len(full_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            num_workers=4, collate_fn=collate_fn)
    
    # Model setup
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    ckpt_dir = "/708HDD/pkc776/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ap50 = 0.0
    
    # Training Loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, grad_accum_steps, scaler=scaler)
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")
        torch.cuda.empty_cache()
        
        ap50 = evaluate(model, val_loader, device, epoch)
        torch.cuda.empty_cache()
        scheduler.step()
        
        # Save best model
        if ap50 > best_ap50:
            best_ap50 = ap50
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
            print(f"New best model saved with AP50: {best_ap50:.4f}")
            
        # Save every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth"))
            print(f"Saved checkpoint for epoch {epoch}")

if __name__ == "__main__":
    main()
