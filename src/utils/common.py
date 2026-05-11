
import os
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def save_checkpoint(model, optimizer,scheduler, epoch, test_acc, test_loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
        "scheduler_state_dict":scheduler.state_dict(),
    }, path)


def setup_data_loaders(args):
    """Setup data transforms and loaders."""
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    assert not (args.train_dir and args.dataset_name), f"Passing both {args.train_dir} and {args.dataset_name} is abigious, use one of them"

    if args.dataset_name:

        from datasets import load_dataset

        train_dataset = load_dataset(args.dataset_name,split="train")
        test_dataset = load_dataset(args.dataset_name,split="test")
    
    elif args.train_dir and args.test_dir :

        from torchvision import datasets

        train_dataset = datasets.ImageFolder(args.train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(args.test_dir, transform=transform)

    else:
        raise ValueError("One of following [dataset_name, [train_dir,test_dir] ] must not be none")

    assert len(train_dataset.classes) == len(test_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    print(f"Found {len(train_dataset.classes)} classes: {train_dataset.classes}")
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    return train_loader, test_loader, train_dataset, test_dataset


def setup_checkpoint_dir(args):
    """Create and setup checkpoint directory."""

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, current_time)
    os.makedirs(checkpoint_dir, exist_ok=False)

    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return checkpoint_dir

def save_results(
    checkpoint_dir,
    best_train_acc,
    best_test_acc,
    all_labels,
    all_preds,
    class_names,
    class_to_idx,
    concepts_report,
    stats: dict,
    ortho_stats:list,
):
    """
    Save training results, confusion matrix, and accuracy/loss plots.
    Expects stats = {'train_acc': [...], 'test_acc': [...], 'train_loss': [...], 'test_loss': [...]}
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)

    result_dict = {
        "best_train_acc": float(best_train_acc),
        "best_test_acc": float(best_test_acc),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "concept_report": concepts_report,
        "ortho_stats":ortho_stats
    }

    with open(os.path.join(checkpoint_dir, "results.json"), "w") as f:
        json.dump(result_dict, f, indent=4)

    # --- Plot Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "confusion_matrix.png"))
    plt.close()

    # --- Plot Accuracy & Loss ---
    train_acc = stats.get("train_acc", [])
    test_acc = stats.get("test_acc", [])
    train_loss = stats.get("train_loss", [])
    test_loss = stats.get("test_loss", [])

    if not all(isinstance(v, (list, tuple)) for v in [train_acc, test_acc, train_loss, test_loss]):
        raise ValueError("Stats dict must contain lists for acc and loss values per epoch.")

    epochs = range(1, len(train_acc) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy subplot
    axs[0].plot(epochs, train_acc, label="Train Accuracy", color="tab:blue", linewidth=2)
    axs[0].plot(epochs, test_acc, label="Test Accuracy", color="tab:orange", linewidth=2)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].set_title("Training vs Testing Accuracy")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # Loss subplot
    axs[1].plot(epochs, train_loss, label="Train Loss", color="tab:green", linewidth=2)
    axs[1].plot(epochs, test_loss, label="Test Loss", color="tab:red", linewidth=2)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Training vs Testing Loss")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.6)

    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(checkpoint_dir, "training_curves.png"))
    plt.close()

    print(f"Results saved to {checkpoint_dir}")
    print(f"cm {cm}")
