import torch

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss/n, correct/n

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    preds, labels = [], []
    for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += loss_fn(out, y).item() * x.size(0)
            n += x.size(0)
            preds.append(out.argmax(1).cpu())
            labels.append(y.cpu())
    preds, labels = torch.cat(preds), torch.cat(labels)
    acc = (preds == labels).float().mean().item()
    return total_loss/n, acc, preds.numpy(), labels.numpy()