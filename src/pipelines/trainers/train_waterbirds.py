import argparse
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from tqdm import tqdm


# -------------------------------
# Dataset wrapper
# -------------------------------
class WaterbirdsDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        # print("row ",row)
        img = row["image"]
        label = row["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------------
# Training Loop
# -------------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += batch_size

        # live updates
        pbar.set_postfix({
            "loss": f"{total_loss/total:.4f}",
            "acc": f"{correct/total:.3f}"
        })

    return total_loss / total, correct / total


# -------------------------------
# Evaluation Loop
# -------------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += batch_size

            pbar.set_postfix({
                "loss": f"{total_loss/total:.4f}",
                "acc": f"{correct/total:.3f}"
            })

    return total_loss / total, correct / total

# -------------------------------
# Main
# -------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    ds = load_dataset("grodino/waterbirds")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_set = WaterbirdsDataset(ds["train"], transform_train)
    test_set = WaterbirdsDataset(ds["test"], transform_eval)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    # Load model dynamically from torchvision
    if not hasattr(models, args.model):
        raise ValueError(f"Model {args.model} not found in torchvision.models")

    model = getattr(models, args.model)(pretrained=True)

    # Replace classifier head
    if "resnet" in args.model:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    else:
        raise NotImplementedError(
            f"Replace your classifier logic for {args.model}"
        )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} "
              f"- Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
              f"- Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Timestamped save dir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = Path(args.save_dir) / timestamp
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path / "checkpoint.pth")
    print(f"Model saved to {save_path}/checkpoint.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to store timestamped checkpoints")
    parser.add_argument("--model", type=str, default="resnet50",
                        help="torchvision model name e.g. resnet50")
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()
    main(args)
