import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from dataset.isic import ISICDataset
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from utils.models import  get_model
import numpy as np


# Configuration
class Config:
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 100
    patience = 20  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "resnet50"

def get_transforms(size:int):

    train_transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform,val_transform



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


def _ensure_binary_vector(logits):
    """
    Safely converts model outputs into a proper (N,) vector of probabilities
    for binary classification.
    """
    # If output is shape (N, 1) → squeeze to (N,)
    if logits.dim() == 2 and logits.size(1) == 1:
        logits = logits.reshape(-1)

    # If output is shape (N,) → fine
    elif logits.dim() == 1:
        pass

    else:
        raise ValueError(
            f"Unexpected output shape {logits.shape}. "
            "Model must output binary logits or class logits."
        )

    return logits


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = _extract_logits(outputs)
        outputs = _ensure_binary_vector(outputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = roc_auc_score(all_labels, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            outputs = _extract_logits(outputs)
            outputs = _ensure_binary_vector(outputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = roc_auc_score(all_labels, all_preds)

    return epoch_loss, epoch_auc

# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, train_auc, val_auc, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_auc': train_auc,
        'val_auc': val_auc
    }
    torch.save(checkpoint, save_path)

# Main training loop
def train_model(train_loader, val_loader, save_dir):
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
    model = get_model(Config.model_name,1,True).to(config.device)
    
    # Binary cross entropy loss
    criterion = nn.BCEWithLogitsLoss()
    
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
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        'device': str(config.device),
        'timestamp': timestamp
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    metrics_history = []
    
    print(f"Training on {config.device}")
    print(f"Saving to: {run_dir}")
    print("-" * 60)
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write(f"Training started at {timestamp}\n")
        f.write(f"Device: {config.device}\n")
        f.write("-" * 60 + "\n")
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, config.device)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Store metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_auc': float(train_auc),
            'val_loss': float(val_loss),
            'val_auc': float(val_auc),
            'learning_rate': float(current_lr)
        }
        metrics_history.append(epoch_metrics)
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{config.num_epochs}\n")
            f.write(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}\n")
            f.write(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}\n")
            f.write(f"Learning Rate: {current_lr:.6f}\n")
        
        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        save_checkpoint(model, optimizer, scheduler, epoch + 1, train_loss, val_loss, 
                       train_auc, val_auc, checkpoint_path)
        
        # Early stopping based on validation loss convergence
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model
            best_model_path = checkpoint_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, scheduler, epoch + 1, train_loss, val_loss,
                          train_auc, val_auc, best_model_path)
            
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
    
    # Final validation
    final_val_loss, final_val_auc = validate(model, val_loader, criterion, config.device)
    print(f"\nFinal Validation AUC: {final_val_auc:.4f}")
    
    with open(log_file, 'a') as f:
        f.write(f"\nFinal Validation AUC: {final_val_auc:.4f}\n")
        f.write(f"Training completed at {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_val_auc': final_val_auc,
        'final_val_loss': final_val_loss
    }, final_model_path)
    
    print(f"\nAll outputs saved to: {run_dir}")
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train ISIC skin lesion classifier')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Torchvision model name')

    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save logs and checkpoints')
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
    
    train_transform,val_transform = get_transforms(299 if "inception" in args.model_name else 224)
    # Create datasets
    train_dataset = ISICDataset(root="/netscratch/aslam/TCAV/ISIC",split='train', transform=train_transform)
    val_dataset = ISICDataset(root="/netscratch/aslam/TCAV/ISIC",split='test', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, 
                             shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    # Train model
    model = train_model(train_loader, val_loader, args.save_dir)