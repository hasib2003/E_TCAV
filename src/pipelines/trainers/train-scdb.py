import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.models import get_model
from .train_ISIC import _extract_logits

# -------------------------
# ARGUMENTS
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("SCDB Training Pipeline")

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model architecture name (e.g. resnet50, inception_v3)",
    )
    parser.add_argument("--data-root", type=str, default="/netscratch/aslam/TCAV/SCDB")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()



# -------------------------
# DATASET
# -------------------------
class SCDB_Dataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.samples = []
        self.labels = []
        self.image_root = image_root
        self.transform = transform

        with open(csv_path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    rel_path, label = line.split("|")
                    label = int(label)
                except ValueError:
                    raise ValueError(
                        f"Malformed line {line_num} in {csv_path}: {line}"
                    )

                self.samples.append(rel_path)
                self.labels.append(label)

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.samples[idx])
        if not os.path.isfile(img_path):
            raise FileNotFoundError(img_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


# -------------------------
# UTILS
# -------------------------
def get_input_size(model_name: str) -> int:
    return 299 if model_name == "inception_v3" else 224


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv = os.path.join(args.data_root, "train.csv")
    val_csv   = os.path.join(args.data_root, "val.csv")
    test_csv  = os.path.join(args.data_root, "test.csv")

    output_dir = os.path.join(args.data_root, f"training_{args.model_name}")
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best.pth")

    input_size = get_input_size(args.model_name)

    train_tfms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = SCDB_Dataset(train_csv, args.data_root, train_tfms)
    val_ds   = SCDB_Dataset(val_csv, args.data_root, eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_classes = len(np.unique(train_ds.labels))

    model = get_model(args.model_name, num_classes, pretrained=True)
    if not model:
        raise ValueError(f"unable to load model")
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(
            train_loader, desc=f"Training [{epoch+1}/{args.epochs}]"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()            
            loss = criterion(_extract_logits(model(images)), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {running_loss / len(train_loader):.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_acc,
                },
                best_model_path,
            )
            print(f"Saved new best model (acc={best_acc:.4f})")

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    main()
