# mlflow-fashion-lab

A small, hands-on project for learning **MLflow** from scratch. It trains a CNN to classify clothing images (Fashion-MNIST) and uses MLflow to track experiments, compare runs and store models.

It is built to be light enough for a normal laptop or a CPU-only server: the dataset is about 30 MB and training takes a few minutes.

> This README is my "re-learn" guide. Follow the steps in order. Each step says what to run, what you should see, and what you learn.

## Contents

1. [What you will learn](#1-what-you-will-learn)
2. [Project structure](#2-project-structure)
3. [Prerequisites](#3-prerequisites)
4. [Setup](#4-setup)
5. [Step 1: Train and test a model without MLflow](#5-step-1-train-and-test-a-model-without-mlflow)
6. [Step 2: Play with the predictions](#6-step-2-play-with-the-predictions)
7. [Step 3: Hello MLflow (fake data)](#7-step-3-hello-mlflow-fake-data)
8. [Step 4: Open the MLflow UI](#8-step-4-open-the-mlflow-ui)
9. [Step 5: MLflow on the real model](#9-step-5-mlflow-on-the-real-model)
10. [Step 6: Compare experiments](#10-step-6-compare-experiments)
11. [MLflow cheat sheet](#11-mlflow-cheat-sheet)
12. [Troubleshooting](#12-troubleshooting)
13. [Roadmap](#13-roadmap)
14. [Git workflow](#14-git-workflow)

---

## 1. What you will learn

| Topic | Where |
|---|---|
| The normal data-scientist workflow: train / validation / test, overfitting, hyperparameters | Step 1 |
| How an image classifier turns a picture into a prediction, and why it fails on unfamiliar photos | Step 2 |
| MLflow basics: experiment, run, parameter, metric, artifact | Step 3 |
| Using the MLflow UI, including on a remote server | Step 4 |
| Applying MLflow to a real PyTorch model | Step 5 |
| Comparing runs to choose the best settings | Step 6 |

## 2. Project structure

```
mlflow-fashion-lab/
├── src/
│   ├── __init__.py          empty file, makes "src" a Python package
│   ├── data.py              loads Fashion-MNIST, splits train/validation/test
│   ├── model.py             the small CNN (SmallCNN)
│   ├── engine.py            train_one_epoch() and evaluate()
│   ├── train_plain.py       Step 1: training WITHOUT MLflow, saves the weights
│   ├── play.py              Step 2: predictions on test images or your own photo
│   ├── mlflow_hello.py      Step 3: MLflow on fake numbers
│   └── train_mlflow.py      Step 5 and 6: training WITH MLflow
├── docs/
│   ├── learning-log.md      my journal: what I did, saw and asked
│   └── github-learning-log.md   how to push, branch and checkout
├── requirements.txt
├── .gitignore
└── README.md
```

Folders created while running (not committed to Git): `data/` (dataset), `models/` (saved weights), `outputs/` (prediction pictures), `mlruns/` (MLflow artifacts), `mlflow.db` (MLflow database).

## 3. Prerequisites

| Tool | Notes |
|---|---|
| Python 3.11 (3.10+ should work) | check with `python --version` |
| Git and a GitHub account | see section 14 |
| An editor such as VS Code | on a remote server, use the Remote-SSH extension |
| About 2 GB free disk | dataset about 30 MB, PyTorch (CPU) about 200 MB |

No GPU is needed.

## 4. Setup

Run these from the project root (the folder containing `src/`).

```bash
# 1. Create and activate a virtual environment
python -m venv .labvenv
source .labvenv/bin/activate          # Windows: .labvenv\Scripts\activate

# 2. Install PyTorch (CPU version, smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest
pip install -r requirements.txt

# 4. Check
python -c "import torch, mlflow; print(torch.__version__, mlflow.__version__)"
```

`requirements.txt`:
```
mlflow>=3.0,<4
scikit-learn
matplotlib
numpy
requests
pillow
```

**Rules for every command below:** run it from the project root, with the virtual environment active, using `python -m src.<name>` (not `python src/<name>.py`), so the `src` imports work.

---

## 5. Step 1: Train and test a model without MLflow

```bash
python -m src.train_plain
```

Expected: 5 lines (one per epoch) with `train_acc`, `val_acc`, losses, then `FINAL TEST accuracy` of roughly 0.90. The weights are saved to `models/fashion_cnn.pt`.

### Concepts

- **The data:** 70,000 grayscale 28x28 images, 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **Three-way split:**
  - Train (55,000): the model learns from these.
  - Validation (5,000): used during training to watch progress and tune settings.
  - Test (10,000): used **once at the end** for an honest score.
- **Overfitting:** the model memorises the training data. Sign: training accuracy keeps rising while validation accuracy stalls or drops.
- **CNN layers:** Conv (detects patterns), ReLU (adds non-linearity), MaxPool (shrinks the image), Linear (produces 10 scores), Dropout (randomly switches off neurons to fight overfitting).
- **Training loop:** feed a batch, compute the loss (how wrong), backpropagate (who is responsible), optimizer step (adjust weights). One pass over the training data is an **epoch**.
- **Hyperparameters:** settings you choose, not learned: learning rate, batch size, epochs, dropout.

### The problem this creates

Results only appear in the terminal. After 10 runs with different settings you cannot remember which gave what. That is what MLflow solves.

## 6. Step 2: Play with the predictions

```bash
python -m src.play                                    # 12 random test images
python -m src.play --image my_photo.png --invert      # your own photo
```

Nothing opens on screen. Pictures are saved to `outputs/` (`random_predictions.png`, `custom_prediction.png`); open them in your editor. On a remote server there is no display, which is why the script uses the `Agg` matplotlib backend and saves files.

### What to look at

- **Green = correct, red = wrong** titles in the grid.
- **High confidence is not correctness.** Example I saw: Pred Shirt (62%), actual Bag.
- **Domain shift:** the model only knows Fashion-MNIST-style images (light item on a dark background, centred, plain). My own photo was predicted correctly as Sneaker **with** `--invert` and wrongly as Bag **without** it. Use `--invert` when your item is dark on a light background.
- The left panel of the custom prediction shows what the model really receives: a 28x28 image.

---

## 7. Step 3: Hello MLflow (fake data)

Learn MLflow on fake numbers before using it on the real model.

```bash
python -m src.mlflow_hello
```

Expected last line: `Done. Run recorded.` Run it a second time after changing `learning_rate` and the run name to get two runs to compare.

### The five words

| Word | Meaning | Example |
|---|---|---|
| **Experiment** | A folder grouping related runs | `hello-mlflow` |
| **Run** | One execution of your script | `first_run` |
| **Parameter** | A setting, logged once | `learning_rate = 0.001` |
| **Metric** | A number, can be logged many times (per `step`) | `fake_loss` at each epoch |
| **Artifact** | A file attached to a run | `note.txt`, a PNG, the model |

### The pattern

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")   # WHERE to store data
mlflow.set_experiment("hello-mlflow")            # WHICH folder
with mlflow.start_run(run_name="first_run"):     # ONE run
    mlflow.log_param("learning_rate", 0.001)     # setting
    mlflow.log_metric("fake_loss", 0.5, step=1)  # result over time
    mlflow.log_text("hello", "note.txt")         # file
```

### Where MLflow keeps things

- **Backend store** (metadata: params, metrics, run info): `mlflow.db`, a SQLite file.
- **Artifact store** (files: images, models): the `mlruns/` folder.

---

## 8. Step 4: Open the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Leave this terminal running, then open **http://127.0.0.1:5000** in your browser.

**On a remote server**, forward the port first:
- VS Code Remote-SSH: open the **Ports** tab, "Forward a Port", enter `5000`.
- Plain terminal, run on your laptop: `ssh -L 5000:localhost:5000 <user>@<server>`.

### Important: choose "Model training"

MLflow 3.x has two views at the top of the left sidebar: **GenAI** and **Model training**. Your runs, parameters and metrics are under **Model training**. The GenAI view (traces, prompts, judges) will look empty.

### What to explore

1. Open the `hello-mlflow` experiment, then the runs table.
2. Click a run: Parameters, Metrics (curves), Artifacts.
3. Tick two runs and click **Compare**.

### Good habits

- Run **one** `mlflow ui` at a time, always from the project root, with the same `--backend-store-uri` your scripts use.
- Stop it with `Ctrl+C`. To kill stray processes: `pkill -f mlflow`.
- If the UI looks empty, check the data directly:
  ```bash
  python - <<'EOF'
  import mlflow
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  print(mlflow.search_runs(experiment_names=["hello-mlflow"]))
  EOF
  ```

---

## 9. Step 5: MLflow on the real model

```bash
python -m src.train_mlflow --epochs 1 --run-name quick-test    # fast check
python -m src.train_mlflow --epochs 5 --run-name baseline
```

Open the UI, choose **Model training**, then the experiment `fashion-mnist-cnn`.

### How it maps to Step 3

| Hello MLflow | Real project (`train_mlflow.py`) |
|---|---|
| `log_param("learning_rate", 0.001)` | `log_params({"lr": ..., "epochs": ..., "dropout": ..., "batch_size": ...})` |
| `log_metric("fake_loss", ..., step=epoch)` | `log_metrics({"train_loss", "val_loss", "train_accuracy", "val_accuracy"}, step=epoch)` |
| (final value, no step) | `log_metrics({"test_loss", "test_accuracy"})` |
| `log_text("hello", "note.txt")` | `log_figure(fig, "confusion_matrix.png")` |
| (new) | `mlflow.pytorch.log_model(model, name="model", ...)` |

### What to look at in the UI

- **Metrics tab:** `train_loss` vs `val_loss`. If train keeps falling while validation rises, the model is overfitting.
- **Artifacts tab, `confusion_matrix.png`:** rows are the true class, columns the predicted class. Look for the largest off-diagonal numbers (usually Shirt / T-shirt / Pullover).
- **Artifacts tab, `model/`:** the model saved in MLflow's standard format, ready for reuse and serving later.

### Training options

```
--epochs      number of passes over the training data (default 5)
--lr          learning rate (default 0.001)
--batch-size  images per step (default 64)
--dropout     dropout probability (default 0.25)
--run-name    label shown in the MLflow UI
```

## 10. Step 6: Compare experiments

Run several variations, changing **one setting at a time**:

```bash
python -m src.train_mlflow --run-name lr-high    --lr 0.01
python -m src.train_mlflow --run-name lr-low     --lr 0.0001
python -m src.train_mlflow --run-name dropout-50 --dropout 0.5
python -m src.train_mlflow --run-name long       --epochs 12
```

In the UI, tick runs and click **Compare**. Questions to answer in `docs/learning-log.md`:

- Which learning rate is best? What does a too-high one look like in the curves?
- Does the 12-epoch run overfit?
- Which run would you deploy, and why?

---

## 11. MLflow cheat sheet

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")   # where
mlflow.set_experiment("name")                    # which folder

with mlflow.start_run(run_name="x"):
    mlflow.log_param("k", v)                     # one setting
    mlflow.log_params({"a": 1, "b": 2})          # many settings
    mlflow.log_metric("loss", 0.3, step=1)       # one number over time
    mlflow.log_metrics({"a": 1.0}, step=1)       # many numbers
    mlflow.log_figure(fig, "plot.png")           # matplotlib figure
    mlflow.log_text("hi", "note.txt")            # text file
    mlflow.log_artifact("file.csv")              # any local file
    mlflow.pytorch.log_model(model, name="model")  # a PyTorch model
```

Read results back in code: `mlflow.search_runs(experiment_names=["fashion-mnist-cnn"])`.

## 12. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root with `python -m src.<name>`. |
| Nothing appears when running `play.py` | No display on a server. Open the PNG in `outputs/`. |
| UI loads but looks empty | Click **Model training** (not GenAI). Then check with `search_runs` (section 8). |
| UI does not open in the browser | Forward the port (section 8). Make sure only one `mlflow ui` is running. |
| `Address already in use` | Another UI is on that port. `pkill -f mlflow` or use `--port 5050`. |
| UI shows different data than your script | Relative path `sqlite:///mlflow.db` depends on the current folder. Start both from the project root. |
| Prediction on my own photo is nonsense | Domain shift. Photograph one item, centred, plain background, and try with and without `--invert`. |
| Slow training | Use `--epochs 1` for tests. |
| Dataset download fails | Needs internet on first run; data is cached in `data/`. |

## 13. Roadmap

- [x] Milestone 1: plain training and testing
- [x] Milestone 1b: play with predictions
- [x] Milestone 2a: hello MLflow
- [ ] Milestone 2c: MLflow on the real model
- [ ] Milestone 3: compare runs
- [ ] Milestone 4: Model Registry (named, versioned models, aliases such as `champion`)
- [ ] Milestone 5: Docker (MLflow server and training in containers, so anyone can run it)
- [ ] Milestone 6: serve the model as a REST API

Tick the boxes as you go.

## 14. Git workflow

Full details are in [`docs/github-learning-log.md`](docs/github-learning-log.md). Quick version:

```bash
git status
git add <files>
git commit -m "Short description"
git push

git switch -c feature/<name>      # new branch
git switch main                   # back to main
git tag <milestone-name> && git push --tags
```

Never commit `data/`, `models/`, `outputs/`, `mlruns/`, `mlflow.db` or the virtual environment (they are in `.gitignore`).