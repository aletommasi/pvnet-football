from __future__ import annotations
import torch
from torch.utils.data import DataLoader, TensorDataset

def make_loader(X, y_shot, y_goal, batch_size=4096, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_shot, dtype=torch.float32),
        torch.tensor(y_goal, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb_shot, yb_goal in loader:
        xb = xb.to(device)
        yb = torch.stack([yb_shot, yb_goal], dim=1).to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    return total_loss / max(n, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits = []
    all_y = []

    for xb, yb_shot, yb_goal in loader:
        xb = xb.to(device)
        yb = torch.stack([yb_shot, yb_goal], dim=1).to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(yb.detach().cpu())

    if all_logits:
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_y = torch.cat(all_y, dim=0).numpy()
    else:
        all_logits, all_y = None, None

    return total_loss / max(n, 1), all_logits, all_y
