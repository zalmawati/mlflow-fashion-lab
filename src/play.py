import argparse
import os

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use("Agg") #run on remote server (no screen needed)
import numpy as np 
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps 
from torchvision import datasets 

from src.data import CLASSES, TRANSFORM 
from src.model import SmallCNN 

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_model(path="models/fashion_cnn.pt"):
    model = SmallCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval() #switch off dropout: we are predicting, not training
    return model 

def predict(model, batch):
    """batch shape: (N, 1, 28, 28) -> predicted class ids, confidences, all probabilities"""
    with torch.no_grad():
        scores = model(batch)  # raw scores ("logits"), one per class
        probs = F.softmax(scores, dim=1)  # turn scores into probabilities that sum to 1
    conf, idx = probs.max(dim=1)
    return idx, conf, probs

def show_random_grid(model, n=12):
    ds = datasets.FashionMNIST("data", train=False, download=True) #raw PIL images
    picks = np.random.choice(len(ds), n, replace=False)
    fig, axes = plt.subplots(3,4,figsize=(9,18))
    for ax, i in zip(axes.flat, picks):
        img, label = ds[i]
        idx, conf, _ = predict(model, TRANSFORM(img).unsqueeze(0)) 
        ok = idx.item() == label 
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Pred: {CLASSES[idx.item()]} ({conf.item():.0%})\nActualL {CLASSES[label]}",
                        fontsize=9, color="green" if ok else "red")
        ax.axis("off")
    plt.tight_layout()
    # plt.show()
    path = f"{OUT_DIR}/random_predictions.png" 
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved random predictions grid to {path}")
    

def show_custom_image(model, path, invert):
    img = Image.open(path).convert("L")        # grayscale
    if invert:
        img = ImageOps.invert(img)             # dark item on light background -> light on dark
    img = ImageOps.autocontrast(img)
    img = ImageOps.fit(img, (28, 28))          # crop to square, shrink to 28x28
    idx, conf, probs = predict(model, TRANSFORM(img).unsqueeze(0))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.imshow(img, cmap="gray")
    a1.set_title("What the model actually sees (28x28)")
    a1.axis("off")
    a2.barh(CLASSES, probs[0].numpy())
    a2.invert_yaxis()
    a2.set_xlim(0, 1)
    a2.set_title(f"Prediction: {CLASSES[idx.item()]} ({conf.item():.0%})")
    plt.tight_layout()
    # plt.show()
    path = f"{OUT_DIR}/custom_prediction.png"  
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved custom prediction to {path}") 

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="path to your own photo (jpg/png)")
    p.add_argument("--invert", action="store_true", help="use if your item is dark on a light background")
    args = p.parse_args() 
    model = load_model()
    if args.image:
        show_custom_image(model, args.image, args.invert)
    else:
        show_random_grid(model)