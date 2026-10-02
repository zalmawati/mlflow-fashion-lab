"""
Milestone 3c: the hello-mlflow ideas applied to the REAL Fashion-MNIST model.

Same 4 MLflow calls you already know:
    set_tracking_uri / set_experiment   -> WHERE and WHICH folder
    start_run                           -> ONE training run
    log_params                          -> settings (once)
    log_metrics(step=epoch)             -> results over time (curves)
Now here:
    log_figure                          -> artifact: confusion matrix picture
    pytorch.log_model                   -> artifact: the trained model itself 
"""

import argparse
import os 

import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt 
import mlflow 
import mlflow.pytorch 
import numpy as np 
import torch 
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix 
from torch import nn 

from src.data import CLASSES, get_loaders 
from src.engine import evaluate, train_one_epoch
from src.model import SmallCNN

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = "cuda" if  torch.cuda.is_available() else "cpu" 

    # same as hello-mlflow: WHERE + WHICH
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("fashion-mnist-cnn")

    # same as plain training
    train_dl, val_dl, test_dl = get_loaders(batch_size=args.batch_size)
    model = SmallCNN(dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # ONE run = everything inside this block
    with mlflow.start_run(run_name=args.run_name):
        #PARAETERS: the settings I chose (logged once)
        mlflow.log_params({
            "epochs": args.epochs,
            "lr":args.lr,
            "batch_size":args.batch_size,
            "dropout": args.dropout,
            "optimizer": "Adam",
            "device": device,
        })

        # METRICS: logged every epoch with step=epoch -> MLflow draws curves
        for epoch in range (1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_dl, loss_fn, optimizer, device)
            va_loss, va_acc, _, _ = evaluate(model, val_dl, loss_fn, device)
            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
            }, step=epoch)
            print(f"Epoch {epoch}: train_acc={tr_acc:.3f} val_acc={va_acc:.3f} "
                  f"train_loss={tr_loss:.3f} val_loss={va_loss:.3f}")

        # FINAL TEST: touched once, logged once (no step)
        te_loss, te_acc, preds, labels = evaluate(model, test_dl, loss_fn, device)
        mlflow.log_metrics({"test_loss": te_loss, "test_accuracy": te_acc})
        print(f"TEST accuracy: {te_acc:.3f}")

        #ARTIFACT 1: confusion matrix picture (like note.txt, but an image)
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(8,8))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
            ax=ax, xticks_rotation=45, colorbar=False)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        #ARTIFACT 2: the trained model, in MLflow's standard format
        # (the input_example lets MLflow record what input shape the model expects)
        model.to("cpu")
        example = np.zeros((1, 1, 28, 28), dtype=np.float32)
        mlflow.pytorch.log_model(model, name="model", input_example=example)

        print("Run logged. Open the MLflow UI -> Model training -> fashion-mnist-cnn")

if __name__ == "__main__":
    main()
    