# GitHub Learning Log (push, branch, checkout)

A reference I can come back to whenever I need to push code, make a branch, or go back to an older version.

Menal model: **Git** keep history on my machine (the server). **GitHub** is a copy of that history online. `push` sends my commits up, `pull` brings other commits down.

```
working folder --git add--> staging area --git commit--> local history --git push--> GitHub
```

---

## 1. One-time setup
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

### 1.2 Make sure I am in the right repo (important on this server)
My project lives inside `~/greenphyto/gp_data_pipeline/...`. A parent folder might be another git repo. Check:
```bash
cd ~/greenphyto/gp_data_pipeline/ml_ops_exploration/mlflow-fashion-lab
git rev-parse --show-toplevel
```

The output **must end with `mlflow-fashion-lab`**. If it shows a parent folder, my project is part of a different repo and I must NOT push it to my personal GitHub. Fix with `git init` inside `mlflow-fashion-lab`, which makes it its own repo (but ask whoever owns the parent repo first, and consider adding the folder to the parent's `.gitignore`).

### 1.3 Check `.gitignore` before the first commit
Never commit data, database, virtual environments or run output:
```
.labvenv/
.venv/
data/
models/
outputs/
mlruns/
mlartifacts/
mlflow.db
__pycache__/
*.pyc
```
If I accidentally comitted something already, stop tracking it (the file stays on disk):
```bash
git rm -r --cached mlflow.db mlruns data
git commit -m "Stop tracking generated files"
```
(Note: `.gitignore` only affects files that are not tracked yet.)

### 1.4 Create the empty repository on GitHub
1. GitHub -> **New repository** -> name `mlflow-fashion-lab`.
2. Choose Public or Private.
3. **Do NOT** tick "Add a README", ".gitignore" or "license". Starting empty avoids a conflict on the first push.

### 1.5 Authentication (pick one)
GitHub no longer accepts account passwords for `git push`.

> The server user `gpadmin` looks like a shared account. Anyone who can use that account can use whatever credentials are stored on it. Prefer a token limited to this one repository, and do not store it permanently if others share the account.

**Option A: HTTPS + Personal Access Token (simplest)**
1. GitHub -> Settings -> Developer settings -> Personal access tokens -> **Fine-grained token**.
2. Limit it to the `mlflow-fashion-lab` repository, permission **Contents: Read and Write**, short expiry.
3. When `git push` asks for a password, paste the token (username = my GitHub username).

**Option B: SSH key**
```bash
ssh-keygen -t ed25519 -C "you@example.com"
cat ~/.ssh/id_ed25519.pub  #copy this output
```
GitHub -> Settings -> **SSH and GPG keys** -> New SSH key -> paste. Then test:
```bash
ssh -T git@github.com
```

### 1.6 Connect and push for the first time
```bash
git remote add origin https://github.com/<username>/mlflow-fashion-lab.git
# or for SSH: git@github.com:<username>/mlflow-fashion-lab.git
git remote -v  # verify
git branch -M main # make sure the branch is called main
git push -u origin main  # -u remembers origin/main for next time
```
After this, a plain `git push` / `git pull` works on `main`.

---

## 2. Everyday workflow
```bash
git status  # what changed? what is staged?
git add src/mlflow_hello.py docs/learning-log.md   # stage specific files (safer than "git add .")
git diff --staged  # review exactly what will be committed
git commit - m "Add hello-mlflow script and learning log"
git push
```
Other useful commands:
```bash
git  log --oneline --graph --decorate -15  # compact  history
git pull  # fetch and merge new commits from GitHub
git restore src/play.py  # throw away UNcommitted edits in a file
git restore --stagged src/play.py  #un-stage a file (keeps my edits)
```

**Commit message habit:** short, present tense, says what changed. Example: `Milestone 2: track Fashion-MNIST training with MLflow`.

---

## 3. Branches

A **branch** is a separate line of work. `main` always stays working, and I experiment on branches.

```bash
git branch # list local branches (* = current)
git switch -c feature/m2-mlflow-training # create AND switch to a new branch
# ...edit, add, commit as usual ...
git push -u origin feature/m2-mlflow-training  # first push of this branch 
git switch main  # go back to main (my uncommitted work muse be committed or stashed first)
git switch feature/m2-mlflow-training  # go back to the feature branch 
```
`git checkout <branch>` does the same as `git switch <branch>`. `switch` is the newer, clearer command.

### Merging a branch back into main

**Via GitHub (recommended, and good practice):**
1. Push the branch.
2. On GitHub click **Compare & pull request** -> **Create pull request** -> **merge"**.
3. Update the local copy:
```bash
git switch main
git pull
git branch 0d feature//m2-mlflow-training  # delete the local branch
git push origin --delete feature/m2-mlflow-training  # optional: delete the remote branch
```

**Locally:**
```bash
git switch main
git merge feature/m2-mlflow-training 
git push
```

### Branch names I will use
`feature/m2-mlflow-training`, `feature/m3-compare-runs`, `feature/m5-docker`, `fix/ui-port`

### Stash: park unfinished work
```bash
git stash            # hide uncommitted changes
git switch main
git stash pop        # bring them back
```

---

## 4. Checking out older versions and tags

Tag each milestone so I can return to it:
```bash
git tag m1-plain-training
git tag m2a-hello-mlflow
git push --tags
git tag  # list tags
```
Look at an old state (read-only exploration):
```bash
git log --oneline 
git switch --detach m1-plain-training  # or a commit id like a1b2c3d
```
This is **detached HEAD**: I am looking at the past and any commit I make here belongs to no branch. To come back: `git switch main`. To keep work started from an old point: `git switch -c experiment-from-m1`.

Get one old file without mocing anything:
```bash
git restore --source=m1-plain-training src/train_plain.py
```

---

## 5. Common problems

| Message / situation | What it means and the fix |
|---|---|
| `Authentication failed` | Password was used instead of a token, or the token expired or lacks Contents write permission. Create a new token. |
| `rejected ... (fetch first)` / `non-fast-forward` | GitHub has commits I do not have (for example a README created on the website). Run `git pull --rebase`, then `git push`. |
| `refusing to merge unrelated histories` | I started the GitHub repo with a README and also had local commits. Run `git pull origin main --allow-unrelated-histories`. |
| Merge conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in a file | Both sides changed the same lines. Edit the file to the final text, remove the markers, then `git add <file>` and `git commit`. |
| `You are in 'detached HEAD' state` | Looking at an old commit. `git switch main` to return. |
| `fatal: not a git repository` | Wrong folder. `cd` to the project root. |
| Huge push or file over 100 MB rejected | I committed data or `mlruns/`. Remove with `git rm -r --cached`, update `.gitignore`. If it is already in history, ask me for help before pushing. |
| `git add .` added junk | `git restore --staged .` then add files individually. |

---

## 6. Cheat sheet

```bash
# see
git status | git log --oneline --graph -15 | git diff | git branch -a | git remote -v
# save
git add <file> | git commit -m "msg" | git push | git pull
# branch
git switch -c <new> | git switch <name> | git merge <name> | git branch -d <name>
# undo (safe)
git restore <file> | git restore --staged <file> | git revert <commit>
# tags
git tag <name> | git push --tags | git switch --detach <tag>
```

## 7. Push log (my own notes)

| Date | Branch | What I pushed | Notes |
|---|---|---|---|
| | main | hello-mlflow + learning logs | first push |
