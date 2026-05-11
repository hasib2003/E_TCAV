import torch
import torch.nn as nn
import torch.optim as optim

import os



from utils.models import load_from_checkpoint, get_model, freeze_backbone
from utils.train import test, train_epoch
from utils.common import setup_checkpoint_dir,setup_data_loaders,save_checkpoint,save_results


from pipelines.args.train import args_train_base

def main():
    
    args = args_train_base()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print config
    print(f"{'='*20} Config {'='*20}")
    args_dict = vars(args)

    # compute max key length for neat alignment
    max_key_len = max(len(k) for k in args_dict.keys())

    for key, val in args_dict.items():
        print(f"{key:<{max_key_len + 3}}: {val}")

    print(f"{'='*50}")


    # Setup
    checkpoint_dir = setup_checkpoint_dir(args)
    train_loader, test_loader, train_dataset, test_dataset = setup_data_loaders(args)
    
    num_classes = len(train_dataset.classes)
    model = get_model(args.model, num_classes,args.pretrained)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint if specified
    if args.resume_checkpoint:
        assert os.path.isfile(args.resume_checkpoint), f"Invalid checkpoint path"
        chkpnt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(chkpnt["model_state_dict"])
        optimizer.load_state_dict(chkpnt["optimizer_state_dict"])
        print(f"weights loaded successfully")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.freeze_backbone:
        model = freeze_backbone(model)

   # Setup checkpoint paths
    checkpoint_paths = {
        'best': os.path.join(checkpoint_dir, "best_model.pth"),
        'last': os.path.join(checkpoint_dir, "last_model.pth"),
        'best_acc': 0.0
    }


    # Training loop
    for epoch in range(args.epochs):
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                               optimizer, device)
        
        test_loss, test_acc = test(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}%")

        # Save checkpoints
        save_checkpoint(model, optimizer, epoch, test_acc, test_loss, 
                       checkpoint_paths['last'])

        if test_acc > checkpoint_paths['best_acc']:
            checkpoint_paths['best_acc'] = test_acc
            save_checkpoint(model, optimizer, epoch, test_acc, test_loss, 
                          checkpoint_paths['best'])
            print(f"Saved best model (Best Acc: {test_acc:.4f}%)")

    print(f"\nTraining complete! Best test accuracy: {checkpoint_paths['best_acc']:.4f}%")

    # Final evaluation
    print(f"\nTesting on test dataset")
    print(f"loading best model ...")

    model = load_from_checkpoint(model, path=checkpoint_paths['best'], device=device)
    test_loss, test_acc, all_labels, all_preds = test(
        model, test_loader, criterion, device, return_preds=True
    )


    save_results(checkpoint_dir, train_acc, test_acc, all_labels, all_preds,
                test_dataset.classes, test_dataset.class_to_idx,{})


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()