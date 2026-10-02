# MLflow Learning Log

My journey learning MLflow with a small image-classification project (Fashion-MNIST).
One entry per milestone. Format: What I did, what I observed, what confused me, and my questions.

Environment: remote linux server (no screen), Python venv `.labvenv`, Mlflow 3.16.1, PyTorch (CPU).

---

## Milestone 1: Setup and Roadmap
- **What I did:** created `mlflow-fashion-lab`, a python venv, installed PyTorch (CPU), Mlflow, scikit-learn, matplotlib. Initialised git.
- **Roadmap:** (1) plain training, (2) MLflow tracking, (3) compare runs, (4) model registry, (5) Docker, (6) server the model as an API.
- **Why Fashion-MNIST:** 70,000 small 28x28 grayscale images, about 30 MB, trains on a CPU in minutes.

---

## Milestone 2a: Train and test a model the normal data-scientist wat (no MLflow)
- **What I did:** trained a small CNN (`SmallCNN`) with `python -m src.train_plain`.
- **Concepts I learned:**
  - Three-way split: **train** (learn), **validation** (true and watch for overfitting), **test** (touch once at the end).
  - **Overfitting:** train accuracy keeps rising while calidation accuracy stalls or drops.
  - CNN building blocks: Conv (finds patterns), ReLU (non-linearity), MaxPool (shrinks), Linear (scores), Dropout (reduces overfitting).
  - Trainin loop: batch -> prediction -> loss -> backpropagation -> optimizer step; one **epoch** = one pass over the training data.
  - **Hyperparameters** are settings I choose (learning rate, batch size, epochs, dropout), **parameters** are learned from data (weights & biases).
- **My results:** final test accuracy = `0.899`
- **Pain point that motivates MLflow:** results are only printed in the terminal. After many runs I cannot remember which settings gave which accuracy.

---

## Milestone 2b: Playing with predicstion (`src/play.py`)
- **What I did:** saved the trained weights to `models/fashion_cnn.pt` and wrote a script that predicst on random test images on my own photo.
- **Problem and fix:** the script printed nothing. My server has no display, so `plt.show()` does nothing. Fix `matplotlib.use("Agg")` and `plt.savefig(---)`, then open the PNG in VS Code.
- **What I observed:** 
    - My own photo **with `--invert`**: predicted **Sneakers** (correct).
    - The same photo **without `--invert`**: predicted **Bags** (wrong).
    - One red example in the random grid: predicted **Shirt (62%)** but the actual label was **Bag**.
- **Lessons:**
    1. **Domain shift:** Fashion-MNIST items are light on a dark background and centred. If m photo does not look like the training data, the model guesses badly.
    2. **Confidence is not correctness:** 62% confident and still wrong, which is why we measure accuracy on a test set.
    3. The model only sees numbers (28x28 pixel values), then 10 scores, thes softmax turns them into probabilities.

---

## Milestone 3a: "Hello MLflow" (`src/mlflow_hello.py`)
- **What I did:** logged fake data to understand MLflow before touching the real model.
- **Core vocabulary:**
| Word | Meaning | Example in my script |
|---|---|---|
| Experiment | A folder of related runs | `hello-mlflow` |
| Run | One execution | `first_run`, `second_run` |
| Parameter | A setting, logged once | `learning_rate`, `epochs` |
| Metric | A number logged over time (uses `step`) | `fake_loss` per epoch |
| Artifact | A little attached to a run | `note.txt` |

- **Where data lives:**
    - Backend store (metadata: params, metrics) = `mlflow.db` (SQLite),
    - Artifact store (files) = `mlrund/` folder.
- **Code pattern to remember:**
    ```python 
    mlflow.set_tracking_uri("sqlite:///mlflow.db") # WHERE to store 
    mlflow.set_experiment("hello-mlflow")          # WHICH folder
    with mlflow.start_run(run_name="first_run") :  # ONE rune
        mlflow.log_param("learning_rate", 0.001)     
        mlflow.log_metrics("fake_loss", 0.5, step=1)
        mlflow.log_text("hello", "note.txt")

---

## Milestone 3b: Debugging the "empty" MLflow UI 
- **Symptom:** the UI opened in my browser but `hello-mlflow` looked empty.
- **First gues (wrong):** the UI was reading a different database or an old process was on port 5000.
- **Check I ran (still usefull):**
    - `ls-la mlflow.db` -> the file exists (about 876 KB).
    - `mlflow.search_runs(...)` -> both runs are in the database.
    - `ps aux | grep -i "[m]lflow"` -> one UI process, already using the correct database.
    - `curl .../api/2.0/mlflow/experiments/search` -> the server returns `hello-mlflow`.
- **Actual cause:** MLflow 3.x UI has two views at the top of the sidebar:  **GenAI** and **Model training**. I was on **GenAI** (traces, prompts, judges), which is empty for this script. Runs and metrics are under **Model training**.
- **Lessons:**
    1. Check the data directly (search_runs or curl) before debugging the server.
    2. Run **one** `mlflow ui` at a time, started with the same ``--backend-store-uri` as the scripts.
    3. On a remote server the UI needs **port forwarding** (VS Code ports tab or `ssh -L`), but if not also OK.
    4. Relative SQLite paths depend on the folder U run from. Always run from project root. 

---

## Milestone 3c: MLflow on the real model
