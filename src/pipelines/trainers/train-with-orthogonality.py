import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
from tqdm import tqdm

from utils.models import load_from_checkpoint, get_model, freeze_backbone, unfreeze
from utils.train import test, train_epoch
from utils.captum import get_concept_significance, get_CAV

from utils.common import setup_checkpoint_dir, save_checkpoint, save_results, setup_data_loaders
from utils.logger import Logger

from pipelines.args.train import args_train_with_concepts
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])


def optimizer_to(optim, device):
    for param in optim.state.values():
        for k, v in param.items():
            if isinstance(v, torch.Tensor):
                param[k] = v.to(device)


def scheduler_to(scheduler, device):
    """Move scheduler state to device if needed."""
    if hasattr(scheduler, 'state_dict'):
        for key, value in scheduler.state_dict().items():
            if isinstance(value, torch.Tensor):
                value.to(device)


def get_target_concepts(path: str):
    """Load target concepts from json file path."""
    concepts = None
    with open(path, 'r') as f:
        concepts = json.load(f)
    
    return concepts


def generate_concept_alignment_report(model, target_concepts, activation_layers: list[str], classifier, concepts_dir, device):
    reports = []

    for concept in target_concepts:
        concept_report = get_concept_significance(
            model=model,
            layers=activation_layers,
            classifier=classifier,
            concepts_dir=concepts_dir,
            concept_name=concept["name"],
            class_idx=concept["idx"],
            eval_images_path=concept["eval_dir_path"],
            trainable=False,
            score_type="sign_count",
            device=device
        )

        reports.append({"info": concept, "results": concept_report})
    
    return reports


def orthogonalize_row_grad(param: torch.nn.Parameter, row_idx: int, v: torch.Tensor, eps: float = 1e-8):
    """
    Modify param.grad in-place so that the gradient of the row `row_idx` is orthogonal to v.
    - param: 2D parameter (e.g. linear.weight) shape (out_features, in_features)
    - row_idx: index of the row to orthogonalize
    - v: 1D tensor with length == in_features
    """
    cosine_sim = torch.nn.functional.cosine_similarity(param.grad[row_idx].unsqueeze(0), v.unsqueeze(0)).cpu().item()
    print(f"Pre {cosine_sim=}")
    if param.grad is None:
        print(f"param.grad is none")
        return

    g = param.grad[row_idx]           # shape: (in_features,)
    v = v.to(g.device).type_as(g)     # ensure same dtype/device

    v_norm = torch.norm(v)

    v = v / v_norm  # ensure that v is a unit vector

    if v_norm.item() <= eps:
        # v is (nearly) zero vector — nothing to remove
        print(f"CAV's norm is almost zero")
        return

    proj = torch.dot(g, v) * v           # scalar
    param.grad[row_idx] = g - proj    # in-place replacement (assign)

    cosine_sim = torch.nn.functional.cosine_similarity(param.grad[row_idx].unsqueeze(0), v.unsqueeze(0)).cpu().item()
    print(f"Post {cosine_sim=}")

def orthogonalize_row_weight(layer: torch.nn.Module, row_idx: int, v: torch.Tensor, eps: float = 1e-8):
    """
    Modify param.data in-place so that the row `row_idx` is orthogonal to v.
    - param: 2D parameter (e.g. linear.weight) shape (out_features, in_features)
    - row_idx: index of the row to orthogonalize
    - v: 1D tensor with length == in_features
    - eps: threshold for considering v as zero vector
    """
    cosine_sim = torch.nn.functional.cosine_similarity(layer.weight.data[row_idx].unsqueeze(0), v.unsqueeze(0)).cpu().item()
    print(f"Pre {cosine_sim=}")
    if layer.weight.data is None:
        print(f"param.grad is none")
        return
    
    with torch.no_grad():  # Don't track this operation in autograd
        
        w = layer.weight.data[row_idx]           # shape: (in_features,)
        v = v.to(w.device, dtype=w.dtype) # ensure same dtype/device
        
        # Check if v is nearly zero BEFORE normalization
        v_norm = torch.norm(v)
        if v_norm <= eps:
            print(f"Warning: CAV's norm ({v_norm.item():.2e}) is nearly zero, skipping orthogonalization")
            return
        
        # Normalize v to unit vector
        v_unit = v / v_norm
        
        wv = torch.dot(w, v_unit)
        proj = wv * v_unit
        
        # Orthogonalize in-place
        layer.weight.data[row_idx] = w - proj
        # layer.weight.data[0] = torch.zeros_like(layer.weight.data[0])
        # layer.weight.data[1] = torch.zeros_like(layer.weight.data[1])

        cosine_sim = torch.nn.functional.cosine_similarity(layer.weight.data[row_idx].unsqueeze(0), v.unsqueeze(0)).cpu().item()
        print(f"Post {cosine_sim=}")

        return layer

def train_ortho_epoch(
    epoch,
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    target_concepts,
    classifier,
    concepts_dir,
    num_exps=10,
    alpha=0.75,
):
    """
    Train one epoch with orthogonality constraint on one or more rows of the fc layer.
    Returns only Python primitives (floats/ints).
    """
    model = model.to(device)
    optimizer_to(optimizer, device)

    total_loss = 0.0
    correct = 0
    total = 0

    # --- Compute orthogonality constraints (CAVs) ---
    print("Computing CAVs...")

    

    # orthogonality_constraints = {}
    all_cavs = {}

    model.eval()
    for cls_name, concept_config in target_concepts.items():
        print(f"# Processing {cls_name}")
        cavs = get_CAV(
            model,
            classifier=classifier,
            concepts_dir=concepts_dir,
            concept_name=concept_config["concept_name"],
            device=device,
            num_rand_concepts=num_exps,
        )

        all_cavs[cls_name] = cavs["avgpool"][0].detach().to(device)

        # mean_cav = cavs[0].detach().to(device)
        # orthogonality_constraints[concept_config["idx"]] = mean_cav

    
    # Apply orthogonality constraint to FC layer gradients
    # for class_idx, cav_tensor in orthogonality_constraints.items():
    #     orthogonalize_row_weight(model.fc.weight, class_idx, cav_tensor)
    # print("... Initial Hard Orthogonalization Applied ... ")


    # --- Training loop ---
    model.train()
    tq = tqdm(train_loader, desc=f"Epoch {epoch} | Ortho Training", unit=" batch")

    model.to(device)
    for batch_idx, (inputs, targets) in enumerate(tq):

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(inputs)
        loss = criterion(preds, targets)

        # Standard loss tracking
        preds_class = preds.argmax(dim=1)
        batch_total = int(targets.size(0))
        batch_correct = int(preds_class.eq(targets).sum().cpu().item())

        total += batch_total
        correct += batch_correct


        orthogonality_losses = 0

        # for cls_name, cav in all_cavs.items():

        #     scores = cav @ model.fc.weight[target_concepts[cls_name]["idx"]].reshape(-1,1)
        #     positive_sco

        #     orthogonality_losses += max(cav @ ,cav),torch.tensor(0))


        _loss = loss +  alpha * orthogonality_losses
        total_loss += float(_loss.detach().cpu().item())

        # Backprop
        _loss.backward()

        # make the f.c. rows orthogonal to cav_tensor

        optimizer.step()

        # for class_idx, cav_tensor in orthogonality_constraints.items():
        #     orthogonalize_row_weight(model.fc.weight, class_idx, cav_tensor)
        

        # Progress bar info
        batch_acc = batch_correct / batch_total if batch_total > 0 else 0.0
        tq.set_postfix_str(f"Acc {batch_acc:.2f} | Loss BCE {float(loss.cpu().item()):.3f} | Loss Ortho {float(orthogonality_losses.cpu().item()):.3f}")

    avg_loss = total_loss / (batch_idx + 1)
    avg_acc = 100.0 * correct / total if total > 0 else 0.0

    
    model.eval()
    ortho_stats = check_othogonality(model,target_concepts,classifier,concepts_dir,device)

    print(f"... post epoch orthogonality stats ...")
    
    for idx,stat in enumerate(ortho_stats):
    
        print(f"Stat {idx}: {stat}")

    print('\n')


    # Return only pure Python types
    return float(avg_loss), float(avg_acc)


def check_othogonality(
    model,
    target_concepts,
    classifier,
    concepts_dir:str,
    device:str,
    num_exps=10
):
    """
    computes the dot product between target CAVs and FC weights
    """
    
    model.eval()
    model = model.to(device)

    # --- Compute orthogonality constraints (CAVs) ---
    print("Computing CAVs...")
    orthogonality_stats = []

    for cls_name, concept_config in target_concepts.items():
        print(f"# Processing {cls_name}")
        cavs = get_CAV(
            model,
            classifier=classifier,
            concepts_dir=concepts_dir,
            concept_name=concept_config["concept_name"],
            device=device,
            num_rand_concepts=num_exps,
            weight_idx=0
        )
        mean_cav = torch.mean(cavs["avgpool"],dim=0).detach().to(device)

        weight_vec = model.fc.weight[concept_config["idx"]]
        cosine_sim = torch.nn.functional.cosine_similarity(
            mean_cav.unsqueeze(0), weight_vec.unsqueeze(0)
        ).cpu().item()
        orthogonality_stats.append({"cosine_sim":cosine_sim,"info":concept_config})

        print(f"\n{concept_config=}")
        print(f"{cosine_sim=}")
        print("="*50)

    return orthogonality_stats


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

    # if not args.resume_checkpoint:
        # Setup
    checkpoint_dir = setup_checkpoint_dir(args)
    # else:
        # checkpoint_dir = os.path.dirname(args.resume_checkpoint)

    logger = Logger(log_file=os.path.join(checkpoint_dir, "log.txt"))
    logger.enable()
    
    train_loader, test_loader, train_dataset, test_dataset = setup_data_loaders(args)
    
    num_classes = len(train_dataset.classes)
    model = get_model(args.model, num_classes, args.pretrained)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc    = 0
    train_acc = -1
    if args.resume_checkpoint:
        assert os.path.isfile(args.resume_checkpoint), f"Invalid checkpoint path"
        chkpnt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(chkpnt["model_state_dict"])
        optimizer.load_state_dict(chkpnt["optimizer_state_dict"])
        if "scheduler_state_dict" in chkpnt:
            scheduler.load_state_dict(chkpnt["scheduler_state_dict"])
        if "epoch" in chkpnt:
            start_epoch = chkpnt["epoch"] + 1
        best_acc = chkpnt["test_acc"]
        # if "train_acc" in ch 
        train_acc = chkpnt.get("train_acc",-1)
        print(f"Weights loaded successfully from epoch {start_epoch}")

    model = model.to(device)
    optimizer_to(optimizer, device)

    criterion = nn.CrossEntropyLoss()

    # Setup checkpoint paths
    checkpoint_paths = {
        'best': os.path.join(checkpoint_dir, "best_model.pth"),
        'last': os.path.join(checkpoint_dir, "last_model.pth"),
        'best_acc':best_acc
    }

    assert os.path.isfile(args.concept_config_train), f"Invalid path to config file"
    target_concepts = get_target_concepts(args.concept_config_train)

    stats = {
        "train_acc": [],
        "train_loss": [],
        "test_acc": [],
        "test_loss": [],
        "learning_rate": []
    }

    # Training loop
    for epoch in range(0, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # if args.correction_frequency != 0 and (epoch + 1) > args.correction_frequency:
        #     model = freeze_backbone(model)
        #     train_loss, train_acc = train_ortho_epoch(
        #         epoch, model, train_loader, criterion, optimizer,
        #         device, target_concepts, args.classifier, args.concepts_dir
        #     )
        #     model = unfreeze(model)
        # else:
        #     train_loss, train_acc = train_epoch(model, train_loader, criterion, 
        #                                        optimizer, device)

        model.eval()
        for cls_name, concept_config in target_concepts.items():
            print(f"# Processing {cls_name}")
            
            cavs = get_CAV(
                model,
                classifier=args.classifier,
                concepts_dir=args.concepts_dir,
                concept_name=concept_config["concept_name"],
                device=device,
                num_rand_concepts=10,
            )

            layer_cav = cavs["avgpool"].to(device)

            for cav in layer_cav:
                model.fc = orthogonalize_row_weight(model.fc,0,cav)
                # model.fc.weight.data = torch.zeros_like(model.fc.weight.data)


        train_loss  =  0
        train_acc   = -1



        test_loss, test_acc = test(model, test_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        # Log stats
        stats["train_acc"].append(train_acc)
        stats["test_acc"].append(test_acc)
        stats["train_loss"].append(train_loss)
        stats["test_loss"].append(test_loss)
        stats["learning_rate"].append(scheduler.get_last_lr()[0])

        print(f"Train Loss: {train_loss:.2f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}%")

        # Save checkpoints with scheduler state
        save_checkpoint(model, optimizer, scheduler,epoch, test_acc, test_loss, 
                       checkpoint_paths['last'])

        if test_acc > checkpoint_paths['best_acc']:
            checkpoint_paths['best_acc'] = test_acc
            save_checkpoint(model, optimizer, scheduler,epoch, test_acc, test_loss, 
                          checkpoint_paths['best'])
            print(f"Saved best model (Best Acc: {test_acc:.2f}%)")

    print(f"\nTraining complete! Best test accuracy: {checkpoint_paths['best_acc']:.2f}%")

    # Final evaluation
    print(f"\nTesting on test dataset")
    print(f"Loading best model...")

    if os.path.isfile(checkpoint_paths['best']) :
        model = load_from_checkpoint(model, path=checkpoint_paths['best'], device=device)
    else:
        model = load_from_checkpoint(model, path=checkpoint_paths['last'], device=device)


    test_loss, test_acc, all_labels, all_preds = test(
        model, test_loader, criterion, device, return_preds=True
    )

    # Generating concepts report
    if args.concept_config_test and os.path.isfile(args.concept_config_test):
        test_concepts = get_target_concepts(path=args.concept_config_test)
    else:
        print(f"No test concepts passed, defaulting to training config...")
        test_concepts = target_concepts

    if len(args.test_activation_layers) > 0:
        activation_layers = args.test_activation_layers
    else:
        print(f"No activation layers specified for testing, defaulting to training config...")
        activation_layers = args.train_activation_layers

    concepts_report = generate_concept_alignment_report(
        model=model,
        target_concepts=test_concepts,
        activation_layers=activation_layers,
        classifier=args.classifier,
        concepts_dir=args.concepts_dir,
        device=device,
    )

    ortho_stats = check_othogonality(model,target_concepts,args.classifier,args.concepts_dir,device)

    # Save results
    save_results(checkpoint_dir, train_acc, test_acc, all_labels, all_preds,
                test_dataset.classes, test_dataset.class_to_idx, concepts_report, stats,ortho_stats)

    print(f"Saved to .. {checkpoint_dir}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()