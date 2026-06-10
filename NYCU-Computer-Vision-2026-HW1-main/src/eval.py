import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from dataset import get_dataloaders
from model import ModifiedResNet50


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract DataLoaders
    _, _, test_loader, fallback_classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Load model checkpoint
    checkpoint = torch.load(
        args.model_path, map_location=device, weights_only=True
    )
    classes = checkpoint.get("classes", fallback_classes)

    # Initialize and load weights
    model = ModifiedResNet50(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    predictions = []

    # Inference Loop
    with torch.no_grad():
        for images, filenames in tqdm(
            test_loader, desc="Generating Predictions"
        ):
            images = images.to(device, non_blocking=True)

            # Using torch.amp for inference speedup
            with torch.amp.autocast("cuda"):
                outputs = model(images)

            _, preds = outputs.max(1)

            for filename, pred in zip(filenames, preds):
                class_label = classes[pred.item()]
                image_name = os.path.splitext(filename)[0]
                predictions.append(
                    {"image_name": image_name, "pred_label": class_label}
                )

    # Create the submission file
    # Ensure it's located correctly
    os.makedirs(
        (
            os.path.dirname(args.output_csv)
            if os.path.dirname(args.output_csv)
            else "."
        ),
        exist_ok=True,
    )
    df = pd.DataFrame(predictions)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--model_path", type=str, default="./checkpoints/best_model.pth"
    )
    parser.add_argument("--output_csv", type=str, default="./prediction.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    evaluate(args)
