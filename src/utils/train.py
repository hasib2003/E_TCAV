import torch
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    model = model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    tq = tqdm(loader, desc="Training", unit="batch")

    for images, labels in tq:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        total_loss += float(loss.detach().cpu().item())
        preds = outputs.argmax(dim=1)
        correct += int(preds.eq(labels).sum().cpu().item())
        total += int(labels.size(0))

        batch_acc = correct / total if total > 0 else 0.0
        tq.set_postfix_str(f"Acc {batch_acc:.2f} | Loss {float(loss.cpu().item()):.3f}")

    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total if total > 0 else 0.0

    return float(avg_loss), float(avg_acc)


def test(model, loader, criterion, device, return_preds=False):
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        tq = tqdm(loader, desc="Testing", unit="batch")
        for images, labels in tq:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += float(loss.detach().cpu().item())
            preds = outputs.argmax(dim=1)
            correct += int(preds.eq(labels).sum().cpu().item())
            total += int(labels.size(0))

            if return_preds:
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

            batch_acc = correct / total if total > 0 else 0.0
            tq.set_postfix_str(f"Acc {batch_acc:.2f} | Loss {float(loss.cpu().item()):.3f}")

    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total if total > 0 else 0.0

    if return_preds:
        return float(avg_loss), float(avg_acc), all_labels, all_preds

    return float(avg_loss), float(avg_acc)
