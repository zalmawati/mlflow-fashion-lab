import torch
from torch import nn 
from src.data import get_loaders
from src.model import SmallCNN
from src.engine import train_one_epoch, evaluate
import os

def main():
    torch.manual_seed(42)
    device = "cude" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, test_dl = get_loaders()
    model = SmallCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(1,6):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, loss_fn, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_dl, loss_fn, device)
        print(f"Epoch {epoch}: train_acc={tr_acc:.3f}, val_acc={va_acc:.3f} "
              f"train_loss={tr_loss:.3f} val_loss={va_loss:.3f}")

    te_loss, te_acc, _, _ = evaluate(model, test_dl, loss_fn, device)
    print(f"FINAL TEST accuracy: {te_acc:.3f}")

    # save the model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(),"models/fashion_cnn.pt")
    print("Saved weights to models/fashion_cnn.pt")

if __name__ == "__main__":
    main()   