import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from utils.models import get_model
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration
class Config:
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 100
    patience = 20  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "resnet50"
    num_classes = 2  # Will be updated from dataset

def get_transforms(size: int):
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def _extract_logits(outputs):
    """
    Handles models that return:
    - a single tensor
    - (logits, aux_logits) tuple (e.g., inception_v3)
    Always returns ONLY the main logits.
    """
    if isinstance(outputs, tuple):
        outputs = outputs[0]  # main output, ignore aux
    return outputs


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = _extract_logits(outputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Get predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.long().to(device)

            outputs = model(images)
            outputs = _extract_logits(outputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # Get predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(cm, save_path, class_names=None):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                   train_acc, val_acc, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    torch.save(checkpoint, save_path)


# Main training loop
def train_model(train_loader, val_loader, save_dir, num_classes, class_names=None):
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(save_dir) / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    log_file = run_dir / 'training_log.txt'
    metrics_file = run_dir / 'metrics.json'
    
    config = Config()
    config.num_classes = num_classes
    
    model = get_model(Config.model_name, num_classes, True).to(config.device)
    
    # Cross entropy loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler - ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Save config
    config_dict = {
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'num_epochs': config.num_epochs,
        'patience': config.patience,
        'num_classes': num_classes,
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        'device': str(config.device),
        'model_name': Config.model_name,
        'timestamp': timestamp
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    metrics_history = []
    
    print(f"Training on {config.device}")
    print(f"Number of classes: {num_classes}")
    print(f"Saving to: {run_dir}")
    print("-" * 60)
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write(f"Training started at {timestamp}\n")
        f.write(f"Device: {config.device}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write("-" * 60 + "\n")
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, config.device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, config.device)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Store metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'learning_rate': float(current_lr)
        }
        metrics_history.append(epoch_metrics)
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{config.num_epochs}\n")
            f.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")
            f.write(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
            f.write(f"Learning Rate: {current_lr:.6f}\n")
        
        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        save_checkpoint(model, optimizer, scheduler, epoch + 1, train_loss, val_loss, 
                       train_acc, val_acc, checkpoint_path)
        
        # Early stopping based on validation loss convergence
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model
            best_model_path = checkpoint_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, scheduler, epoch + 1, train_loss, val_loss,
                          train_acc, val_acc, best_model_path)
            
            print(f"✓ New best validation loss: {best_val_loss:.4f}")
            with open(log_file, 'a') as f:
                f.write(f"✓ New best validation loss: {best_val_loss:.4f}\n")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            with open(log_file, 'a') as f:
                f.write(f"No improvement for {patience_counter} epoch(s)\n")
        
        with open(log_file, 'a') as f:
            f.write("-" * 60 + "\n")
        
        # Save metrics history
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=4)
        
        # Stop if validation loss has converged
        if patience_counter >= config.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            
            with open(log_file, 'a') as f:
                f.write(f"\nEarly stopping triggered after {epoch+1} epochs\n")
                f.write(f"Best validation loss: {best_val_loss:.4f}\n")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final validation with detailed metrics
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("=" * 60)
    
    final_val_loss, final_val_acc, final_preds, final_labels = validate(
        model, val_loader, criterion, config.device
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    
    # Calculate F1 score (macro)
    f1_macro = f1_score(final_labels, final_preds, average='macro')
    
    # Calculate per-class F1 scores
    f1_per_class = f1_score(final_labels, final_preds, average=None)
    
    print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final F1 Score (Macro): {f1_macro:.4f}")
    print("\nPer-class F1 Scores:")
    for i, f1 in enumerate(f1_per_class):
        class_label = class_names[i] if class_names else f"Class {i}"
        print(f"  {class_label}: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save confusion matrix plot
    cm_plot_path = run_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, cm_plot_path, class_names)
    print(f"\nConfusion matrix saved to: {cm_plot_path}")
    
    # Save detailed classification report
    report = classification_report(final_labels, final_preds, 
                                   target_names=class_names if class_names else None,
                                   digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Write final results to log file
    with open(log_file, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("FINAL EVALUATION ON VALIDATION SET\n")
        f.write("=" * 60 + "\n")
        f.write(f"Final Validation Accuracy: {final_val_acc:.4f}\n")
        f.write(f"Final Validation Loss: {final_val_loss:.4f}\n")
        f.write(f"Final F1 Score (Macro): {f1_macro:.4f}\n\n")
        f.write("Per-class F1 Scores:\n")
        for i, f1 in enumerate(f1_per_class):
            class_label = class_names[i] if class_names else f"Class {i}"
            f.write(f"  {class_label}: {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"\nTraining completed at {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
    
    # Save final metrics to JSON
    final_metrics = {
        'final_val_accuracy': float(final_val_acc),
        'final_val_loss': float(final_val_loss),
        'f1_score_macro': float(f1_macro),
        'f1_scores_per_class': [float(f1) for f1 in f1_per_class],
        'confusion_matrix': cm.tolist()
    }
    
    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_val_acc': final_val_acc,
        'final_val_loss': final_val_loss,
        'f1_score_macro': f1_macro,
        'confusion_matrix': cm.tolist()
    }, final_model_path)
    
    print(f"\nAll outputs saved to: {run_dir}")
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Biased Celeba for hair color prediction')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Torchvision model name')
    parser.add_argument('--dataset', type=str, required=False,default="necktie",
                       help='Dataset to be used necktie or gender')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save logs and checkpoints')
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='unbiased fraction of samples in celeba dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Update config with arguments
    Config.batch_size = args.batch_size
    Config.learning_rate = args.learning_rate
    Config.num_epochs = args.num_epochs
    Config.patience = args.patience
    Config.model_name = args.model_name
    
    train_transform, val_transform = get_transforms(
        299 if "inception" in args.model_name else 224
    )

    if args.dataset == "necktie":
        from dataset.celeba import CelebABlondeNecktieBiased

        train_dataset = CelebABlondeNecktieBiased(split="train",alpha=args.alpha,transform=train_transform)
        val_dataset = CelebABlondeNecktieBiased(split="val",alpha=args.alpha,transform=val_transform)

    if args.dataset == "gender":

        from dataset.celeba import  CelebAGender

        train_dataset = CelebAGender(split="train",transform=train_transform)
        val_dataset = CelebAGender(split="val",transform=val_transform)
    
    # Get number of classes from dataset
    num_classes = len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 2
    class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Train model
    model = train_model(train_loader, val_loader, args.save_dir, 
                       num_classes, class_names)