import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_dataloaders
from model import ModifiedResNet50
import wandb
from torchvision.transforms import v2


def train(args):
    if args.use_wandb:
        wandb.init(project="CVDL-HW1", config=vars(args))

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, _, classes = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_type=args.aug_type,
    )

    # Initialize Model
    model = ModifiedResNet50(num_classes=len(classes), pretrained=True)
    model = model.to(device)

    # Objective and Optimizer configuration
    # Label smoothing helps to act as regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW incorporates weight decay properly
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine Annealing learning rate scheduler with warmup
    warmup_epochs = args.warmup_epochs
    main_epochs = args.epochs - warmup_epochs

    # 建立 scheduler 函數 (Warmup -> Cosine Annealing)
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(current_epoch - warmup_epochs) / float(
                max(1, main_epochs)
            )
            import math

            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 設定 MixUp / CutMix
    mixup_cutmix = None
    if args.use_mixup:
        mixup_cutmix = v2.RandomChoice(
            [
                v2.CutMix(num_classes=len(classes)),
                v2.MixUp(num_classes=len(classes)),
            ]
        )

    best_val_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    # Mixed precision training
    scaler = torch.amp.GradScaler(device.type)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
            dynamic_ncols=True,
        )
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            # Apply MixUp / CutMix on batches
            if mixup_cutmix is not None:
                mixed_images, mixed_labels = mixup_cutmix(images, labels)
            else:
                mixed_images, mixed_labels = images, labels

            optimizer.zero_grad()

            with torch.amp.autocast(device.type):
                outputs = model(mixed_images)
                loss = criterion(outputs, mixed_labels)

            # Scales loss to avoid underflow
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

            # Metric accumulation
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"}
            )

            if args.use_wandb:
                wandb.log({"train/batch_loss": loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                leave=False,
                dynamic_ncols=True,
            ):
                images, labels = images.to(
                    device, non_blocking=True
                ), labels.to(device, non_blocking=True)

                with torch.amp.autocast(device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f} "
            f"Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} "
            f"Acc: {val_epoch_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if args.use_wandb:
            wandb.log(
                {
                    "train/loss": epoch_loss,
                    "train/acc": epoch_acc,
                    "val/loss": val_epoch_loss,
                    "val/acc": val_epoch_acc,
                    "epoch": epoch + 1,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        scheduler.step()

        # Checkpoint saving
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "classes": classes,
                },
                os.path.join(args.save_dir, "best_model.pth"),
            )
            print(
                "=== Saved new best model with Validation Accuracy: "
                f"{best_val_acc:.4f} ==="
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save the best model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of epochs for learning rate warmup",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (e.g., cuda:0, cpu)",
    )
    parser.add_argument(
        "--use_mixup",
        action="store_true",
        default=True,
        help="Enable MixUp and CutMix",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--aug_type",
        type=str,
        default="randaug",
        choices=["none", "colorjitter", "randaug", "autoaug"],
        help="Augmentation type",
    )

    args = parser.parse_args()
    train(args)
