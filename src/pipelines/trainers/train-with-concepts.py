import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
from tqdm import tqdm

from utils.models import load_from_checkpoint, get_model
from utils.train import test, train_epoch
from utils.captum import get_concept_significance
from utils.common import setup_checkpoint_dir,setup_data_loaders,save_checkpoint,save_results


from loss.concept import ConceptLoss
from pipelines.args.train import args_train_with_concepts



def optimizer_to(optim, device):
    for param in optim.state.values():
        for k, v in param.items():
            if isinstance(v, torch.Tensor):
                param[k] = v.to(device)

def get_target_concepts(path:str):
    """Load target concepts from json file path."""

    concepts = None
    with open(path,'r') as f:
        concepts = json.load(f)
    
    return concepts

def generate_concept_alignment_report(model, target_concepts, activation_layers:list[str],classifier,concepts_dir,device):

    reports = []

    for concept in target_concepts:
        concept_report = get_concept_significance(
            model=model,
            layers=activation_layers,
            classifier = classifier,
            concepts_dir=concepts_dir,
            concept_name=concept["name"],
            class_idx=concept["idx"],
            eval_images_path=concept["eval_dir_path"],
            trainable=False,
            score_type="sign_count",
            device=device
        )

        reports.append({"info":concept,"results":concept_report})
    
    return reports

def compute_concept_losses(model, target_concepts, activation_layers,classifier,concepts_dir,device):
    """Compute concept losses for all target concepts."""
    
    concept_loss_fn = ConceptLoss()
    concept_losses = {}
    c_losses = []

    for concept in target_concepts:
        mean_align_val = get_concept_significance(
            model=model,
            layers=activation_layers,
            classifier = classifier,
            concepts_dir = concepts_dir,
            concept_name=concept["name"],
            class_idx=concept["idx"],
            eval_images_path=concept["eval_dir_path"],
            trainable=True,
            score_type="magnitude",
            device=device
        )

        key = f'{concept["name"]}_{concept["idx"]}'
        concept_losses[key] = {
            "raw": mean_align_val.cpu().item().__round__(3),
            "desired": concept["desired"]
        }

        if (concept["desired"] and mean_align_val < 0) or (not concept["desired"] and mean_align_val > 0):
            c_loss = concept_loss_fn(mean_align_val, concept["desired"])
            c_losses.append(c_loss)
            concept_losses[key]["loss"] = c_loss.cpu().item().__round__(3)

    total_c_loss = sum(c_losses) / len(c_losses) if c_losses else 0
    return concept_losses, total_c_loss

def train_epoch_with_concept(epoch, model, train_loader, test_loader, criterion, optimizer, 
                             device, target_concepts,target_activation_layers,classifier,concepts_dir, checkpoint_paths, alpha=0.5):
    """Train one epoch with concept correction."""
    model.train()
    model = model.to(device)
    optimizer_to(optimizer, device)

    total_loss = 0
    correct = 0
    total = 0
    best_test_acc = checkpoint_paths.get('best_acc', 0.0)

    tq = tqdm(train_loader, desc="Training")

    for batch_idx, (images, labels) in enumerate(tq):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Compute concept losses
        concept_losses, total_c_loss = compute_concept_losses(model, target_concepts,target_activation_layers,classifier,concepts_dir, device)

        # print("concept_losses ",concept_losses)

        # Combine losses
        if total_c_loss > 1e-4:
            loss = (1 - alpha) * loss + alpha * total_c_loss
            print("\n==== Concept Losses ====")
            for concept, details in concept_losses.items():
                details_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in details.items())
                print(f"{concept:<15} | {details_str}")
            print(f"Total Concept Loss: {total_c_loss:.4f}")
            print("=" * 35)


        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)

        b_size = labels.size(0)
        b_corr = predicted.eq(labels).sum().item()
        total += b_size
        correct += b_corr

        tq.set_postfix_str(f"Acc: {b_corr/b_size:.4f}   loss: {loss.cpu().item():.4f}")

        # Periodic evaluation and checkpointing
        if batch_idx % 30 == 0:
            test_loss, test_acc = test(model, test_loader, criterion, device)
            print(f"test_acc {test_acc:.4f} \t test_loss{test_loss:.4f}")

            save_checkpoint(model, optimizer, epoch, test_acc, test_loss, 
                          checkpoint_paths['last'])

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                checkpoint_paths['best_acc'] = best_test_acc
                save_checkpoint(model, optimizer, epoch, test_acc, test_loss, 
                              checkpoint_paths['best'])
                print(f"Saved best model (Best Acc: {test_acc:.4f}%)")

    return total_loss / (batch_idx + 1), 100. * correct / total




def main():
    
    args = args_train_with_concepts()
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



    assert len(args.train_activation_layers) == 1, f"Multiple activation layers are not supported in training"

    # Setup
    checkpoint_dir = setup_checkpoint_dir(args)
    train_loader, test_loader, train_dataset, test_dataset = setup_data_loaders(args)
    
    num_classes = len(train_dataset.classes)
    model = get_model(args.model, num_classes)
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

    # Setup checkpoint paths
    checkpoint_paths = {
        'best': os.path.join(checkpoint_dir, "best_model.pth"),
        'last': os.path.join(checkpoint_dir, "last_model.pth"),
        'best_acc': 0.0
    }

    assert os.path.isfile(args.concept_config_train),f"Invalid path to config file"
    target_concepts = get_target_concepts(args.concept_config_train)

    # Training loop
    for epoch in range(args.epochs):
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        if (epoch + 1) % args.correction_frequency == 0:
            train_loss, train_acc = train_epoch_with_concept(
                epoch, model, train_loader, test_loader, criterion, optimizer,
                device, target_concepts, args.train_activation_layers, args.classifier,args.concepts_dir,checkpoint_paths
            )
        else:
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

    # generating concepts report

    if args.concept_config_test and os.path.isfile(args.concept_config_test):
        test_concepts = get_target_concepts(path=args.concept_config_test)

    else:
        print(f"No test concepts passed, defaulting to training config ...")
        test_concepts = target_concepts

    if len(args.test_activation_layers) > 0:
        activation_layers = args.test_activation_layers

    else:
        print(f"No activation layers specified for testing, defaulting to training config ...")
        activation_layers = args.train_activation_layers

    concepts_report = generate_concept_alignment_report(
                                                        model=model,                                                        
                                                        target_concepts= test_concepts,
                                                        activation_layers=activation_layers,
                                                        classifier=args.classifier,
                                                        concepts_dir=args.concepts_dir,
                                                        device=device, 
                                                        save_path=checkpoint_dir)

    # Save results
    save_results(checkpoint_dir, train_acc, test_acc, all_labels, all_preds,
                test_dataset.classes, test_dataset.class_to_idx,concepts_report)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()