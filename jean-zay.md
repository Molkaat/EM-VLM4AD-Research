# EM-VLM4AD Training on Jean Zay H100 GPU Cluster

Complete workflow guide for training/running EM-VLM4AD on Jean Zay's H100 GPUs.
Adapted from the TokCom project guide, updated with everything learned from that
project's troubleshooting history.

---

## Project Reference

```
User:           usx94yc
Project:        mbp (IDRIS ID: 105524/AD011017614)
Account:        mbp@h100
Default QOS:    qos_gpu_h100-t3
H100 partition: gpu_p6
Project path:   /lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD
Data path:      /lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD/data
Local mirror:   D:\Users\moltrabelsi\Desktop\PHD\tokcom\EM-VLM4AD  (adjust as needed)
```

Check your project is active before submitting:
```bash
echo $IDRPROJ    # must print: mbp
idrproj          # must show [default][active]
```

If `$IDRPROJ` is wrong or empty:
```bash
eval $(idrenv -d mbp)   # activate for current session
# or
idrproj -d mbp          # set as default (needs password, takes effect after reconnect)
```

---

## Critical: H100 Job Submission Requirements

**Every job targeting H100 GPUs must include all three of these:**

```bash
#SBATCH --partition=gpu_p6          # H100 partition (NOT gpu_p13, which is V100)
#SBATCH --account=mbp@h100          # Project@gpu_type format
#SBATCH --qos=qos_gpu_h100-t3       # H100-specific QOS
#SBATCH -C h100                     # Hardware constraint — REQUIRED by IDRIS
```

Without `-C h100`, IDRIS will reject every submission with:
```
IDRIS: Account mbp@h100 ----- Job type v100
```
even if partition and QOS are correct. This is an IDRIS backend requirement.

**QOS options for mbp@h100:**
- `qos_gpu_h100-dev` — short jobs, max 2 hours, **much faster queue** — use this for
  any quick diagnostic/debug script, not just "official" testing. If a job doesn't
  need the full budget of `t3`/`t4`, submitting under `dev` can be the difference
  between running in minutes vs. waiting hours behind a busy `t3` queue.
- `qos_gpu_h100-t3` — standard jobs, up to ~20 hours (use for training)
- `qos_gpu_h100-t4` — long jobs, up to ~4 days 4 hours
- `qos_gpu_h100-gc`, `qos_gpu_h100-as` — check
  `sacctmgr show qos format=Name,MaxWall -p | grep h100` for the full current list.

**Interactive session:**
```bash
srun --partition=gpu_p6 --account=mbp@h100 --qos=qos_gpu_h100-t3 -C h100 --gres=gpu:1 --time=00:30:00 --pty bash
```

**Quick test/debug (faster queue):**
```bash
srun --partition=gpu_p6 --account=mbp@h100 --qos=qos_gpu_h100-dev -C h100 --gres=gpu:1 --time=00:20:00 --pty bash
```

---

## CRITICAL: Always use `env -u BASH_ENV` when submitting

```bash
env -u BASH_ENV sbatch --parsable \
  --partition=gpu_p6 --account=mbp@h100 --qos=qos_gpu_h100-t3 -C h100 \
  --gres=gpu:1 --cpus-per-task=16 --time=01:00:00 \
  --job-name="JOBNAME" \
  --output=logs/JOBNAME_%j.out \
  --error=logs/JOBNAME_%j.err \
  --wrap="cd /lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD && source venv/bin/activate && export HF_HOME=\$SCRATCH/.cache/huggingface && export TORCH_HOME=\$SCRATCH/.cache/torch && export TRANSFORMERS_OFFLINE=1 && python3 SCRIPT.py --args"
```

**Why this matters:** if your shell's `$BASH_ENV` points to IDRIS's module-system init
script, `sbatch` inherits your environment by default, and every job silently sources
that script before running your `--wrap` command, sometimes breaking the job in ways
that are hard to diagnose (empty logs, jobs that fail instantly with no clear error).
`env -u BASH_ENV` strips this before submission. This caused repeated silent failures
in the TokCom project before we identified it — always include it.

**Do NOT add `module purge; module load python/...` lines to your `--wrap` string**
unless you know you specifically need the system Python module. If you use your own
venv (which you should), loading a system Python module on top of it can break the
venv's own package resolution. If you see errors tracing back to a module load line,
remove it — this fixed a recurring `torch_shm_manager` error in the TokCom project.

---

## Quick Reference

### First Time Setup
```powershell
# From your Windows laptop
cd D:\Users\moltrabelsi\Desktop\PHD\tokcom\EM-VLM4AD
# (adapt/create a transfer script similar to TokCom's transfer_to_jean_zay.ps1 if needed)
```

### SSH — set up key-based auth once, to avoid typing your password every command
```powershell
ssh-keygen -t ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh usx94yc@jean-zay.idris.fr "cat >> ~/.ssh/authorized_keys"
ssh usx94yc@jean-zay.idris.fr   # should now connect without a password prompt
```

### Regular Training
```bash
# From Jean Zay login node
cd /lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD
sbatch stage1.sh
tail -f logs/stage1_*.out
```

**Note on `tail -f` showing nothing:** Python buffers stdout when writing to a file
instead of a terminal. A long-running job can show an empty `.out` file for a long
time even while genuinely training — check `nvidia-smi` on the compute node
(`ssh <node> nvidia-smi`, from `squeue`'s NODELIST column) to confirm real GPU
utilization if the log looks stuck. For future scripts, use `python3 -u` instead of
`python3` to force unbuffered output and get genuine real-time logs.

---

## Step-by-Step Setup (First Time Only)

### Step 1: Transfer files to Jean Zay

From Windows, with key-based auth set up, no repeated password prompts:
```powershell
cd D:\Users\moltrabelsi\Desktop\PHD\tokcom\EM-VLM4AD
scp -r .\*.py usx94yc@jean-zay.idris.fr:/lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD
scp -r .\modules usx94yc@jean-zay.idris.fr:/lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD
scp -r .\utils usx94yc@jean-zay.idris.fr:/lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD
scp -r .\data usx94yc@jean-zay.idris.fr:/lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD
```
Watch for the `scp` trailing-backslash trap: `"D:\path\to\dest\"` (backslash right
before the closing quote) breaks Windows' scp, causing `Invalid argument`. Drop the
trailing backslash: `"D:\path\to\dest"`.

Transferring the full `checkpoints/` or `models/` directories later, once created,
follows the same pattern — just expect it to take longer (multi-GB files).

### Step 2: SSH to Jean Zay
```bash
ssh usx94yc@jean-zay.idris.fr
```

### Step 3: Set up the Python environment
```bash
cd /lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --break-system-packages
pip install -r requirements.txt --break-system-packages
```
**Always use `--break-system-packages` with pip on Jean Zay**, or installs will fail
or silently misbehave.

### Step 4: Verify
```bash
df -h $SCRATCH
idr_quota_user
echo $IDRPROJ        # should print: mbp
idrproj              # should show mbp [default][active]
ls -lh /lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD/
python3 -c "import torch, numpy, transformers; print(torch.__version__, numpy.__version__, transformers.__version__)"
```

---

## Correct SLURM Script Header

```bash
#!/bin/bash
#SBATCH --job-name=emvlm4ad_stage1
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_p6
#SBATCH --account=mbp@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH -C h100
```

To verify any script has all required directives:
```bash
grep -E "partition|account|qos|-C h100" stage1.sh
```

---

## Training Workflow

### Submit
```bash
cd /lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD
source venv/bin/activate
mkdir -p logs
sbatch stage1.sh
```

### Chain dependent jobs correctly
```bash
JOB1=$(env -u BASH_ENV sbatch --parsable stage1.sh)
JOB2=$(env -u BASH_ENV sbatch --parsable --dependency=afterok:$JOB1 stage2.sh)
```
**Important:** this only works reliably if `$JOB1` is submitted and stays live in the
*same shell session* as the dependent submission. If Stage 1 already finished and
aged out of SLURM's active job table before you submit Stage 2 with
`--dependency=afterok:$JOB1`, you'll get `Job dependency problem` and the dependent
job will show `DependencyNeverSatisfied` and never run — submit both in one go,
not across separate sessions.

### Monitor
```bash
squeue -u usx94yc
squeue --start -u usx94yc                    # estimated start times
tail -f logs/stage1_*.out
sacct -u usx94yc --starttime=today -o JobID,JobName,State,ExitCode,Elapsed
```

### Cancel
```bash
scancel 883045         # specific job
scancel -u usx94yc     # all your jobs
```

---

## Common Failure Modes (from TokCom project experience)

### "Job type v100" rejection
**Cause:** missing `-C h100`. **Fix:** add `-C h100` to the sbatch command/script.

### "Disk quota exceeded" (Errno 122)
```bash
idr_quota_user
df -h $SCRATCH
pip install package --no-cache-dir --break-system-packages
```
Check both your home directory quota (`~/.local`) and your project/scratch quota —
they're tracked separately and either can block an install.

### Job stuck in pending
- `(Priority)` = normal queue contention, will run eventually. Check
  `squeue --start -u usx94yc` for an estimate, or resubmit under
  `qos_gpu_h100-dev` if the job is short enough.
- `(QOSGrpGRES)` = you've hit a group-level GPU resource cap for that QOS (too many
  jobs running simultaneously under the same account/QOS). Wait for other jobs to
  finish, or reduce how many you submit at once.
- `(DependencyNeverSatisfied)` = the job you depended on already finished/failed and
  aged out before the dependency was registered. Cancel and resubmit both jobs
  together in one session.

### SPICE / pycocoevalcap crashes during eval
If using COCO-style metrics (BLEU/ROUGE/METEOR/CIDEr) via `pycocoevalcap`, calling
`coco_eval.evaluate()` directly will try to compute SPICE, which needs to download
Stanford CoreNLP models or a `spice-1.0.jar` — compute nodes have no internet access,
so this crashes after the (potentially slow) generation step has already completed.
**Fix:** don't call the library's default `evaluate()`. Instead, manually score with
individual scorers, skipping SPICE:
```python
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# tokenize gts/res, then loop scorers = [(Bleu(4), [...]), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]
```
Since predictions/annotations are saved to disk *before* the metrics step, if a job
does crash on SPICE, you don't need to rerun inference — just recompute metrics from
the saved `predictions_*.json`/`annotations_*.json` files directly (seconds, no GPU
needed).

### Venv corruption after a Jean Zay maintenance window
Cluster-wide maintenance can leave a venv partially broken (missing binaries like
`torch_shm_manager`, missing submodules like `numpy._globals`, `ModuleNotFoundError`
for packages that were previously working). Symptoms show up as import errors that
weren't there the day before, for no code-side reason.
**Fix:** if a spare/backup venv exists (`venv_new` or similar), verify it works and
switch to it:
```bash
venv_new/bin/python3 -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
# if healthy:
mv venv venv_broken_$(date +%Y%m%d)
mv venv_new venv
sed -i 's|venv_new|venv|g' venv/bin/activate venv/pyvenv.cfg
source venv/bin/activate
pip check
```
If no backup exists, rebuild from scratch: recover the package list via
`pip freeze` (cross-check against the actual site-packages directory listing, since
`pip freeze` can silently omit packages with damaged metadata), build a fresh venv,
reinstall torch/torchvision from the correct CUDA index, then the rest.

### Cascading version conflicts after fixing one package
Fixing one broken package (e.g. reinstalling `numpy`) can reveal the next
incompatible package once the import gets further (torch → transformers →
huggingface_hub → tokenizers → torchvision, each can be a separate wall).
**General approach:** fix one error at a time, re-test the full import chain after
each fix, and prefer pinning to *known-compatible version ranges* rather than always
grabbing latest — check the error message itself, which frequently states the exact
required range (e.g. `huggingface-hub>=0.19.3,<1.0 is required`).
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from modules.foundation_model import Qwen2ForVQA   # or your equivalent model wrapper
print('Import OK')
"
```
Run this test via a quick `dev`-QoS SLURM job, not directly on the login node — the
login node will `Killed` a large-model load for exceeding its resource limits, which
looks like a crash but isn't one.

### VRAM out of memory
```python
model.gradient_checkpointing_enable()
# or reduce batch size / add gradient_accumulation_steps
```

### Model not found on GPU node
GPU/compute nodes have no outbound internet access. Any model/dataset download must
happen on the login node first (or be pre-transferred), with
`export TRANSFORMERS_OFFLINE=1` set for the actual training/eval job so it doesn't
even attempt a network call.

---

## Key Environment Variables (always export these in your `--wrap` string)
```bash
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

---

## File Structure

```
/lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD/
├── data/                          # dataset files
├── models/                        # downloaded model weights
├── modules/                       # model/architecture code
├── utils/                         # dataset loaders, collators
├── checkpoints/                   # trained checkpoints (organize by stage/variant)
├── results/                       # eval outputs (predictions, annotations, metrics)
├── logs/                          # SLURM job output/error logs — clear periodically:
│                                     find logs/ -type f -mtime +3 -delete
├── stage1.sh, stage2_*.sh         # SLURM training scripts
├── requirements.txt
└── venv/                          # Python virtual environment
```

**Keep experimental variants in clearly separated subfolders** (own checkpoints/
results/logs, self-contained scripts with paths pointing only within that
subfolder). This avoids accidentally overwriting a validated checkpoint or result
set with a new experiment that happens to reuse the same filename — this has
happened more than once and cost real results. Before running a new experiment
line, decide and write down (e.g. in a short `VERSION_LOG.md`) where its outputs
go, and double check no path collides with anything already validated.

---

## File Transfer Reference

```powershell
# Single file
scp file.py usx94yc@jean-zay.idris.fr:/lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD

# Whole directory
scp -r localdir usx94yc@jean-zay.idris.fr:/lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD

# Pulling results back down (no trailing backslash before the closing quote!)
scp -r usx94yc@jean-zay.idris.fr:/lustre/fsn1/projects/rech/mbp/usx94yc/EM-VLM4AD/results "D:\Users\moltrabelsi\Desktop\PHD\tokcom\EM-VLM4AD"
```

With SSH keys set up, none of these prompt for a password.

---

## Useful Commands Reference

```bash
# Account / project
echo $IDRPROJ
idrproj
idrenv

# Cluster info
sinfo -p gpu_p6
squeue -p gpu_p6 | wc -l              # rough measure of current H100 queue congestion
sacctmgr show qos format=Name,MaxWall -p | grep h100   # available QOS options

# Your jobs
squeue -u usx94yc
squeue --start -u usx94yc
sacct -u usx94yc --starttime=today -o JobID,JobName,State,ExitCode,Elapsed
sbatch stage1.sh
scancel 883045
scancel -u usx94yc
scontrol show job 883045

# Storage
idr_quota_user
df -h $SCRATCH
du -sh directory/

# Watching a specific compute node's GPU usage (get node name from squeue NODELIST)
ssh <nodename> nvidia-smi
```

---

## Security Notes

- Never commit Hugging Face tokens to git.
- Store token in `~/.huggingface/token` or `export HF_TOKEN=...`.
- SCRATCH is shared; keep credentials in home directory only.

---

## Getting Help

1. Check logs: `cat logs/stage1_*.out` and the matching `.err` file — errors are
   almost always in `.err`, not `.out`.
2. Check quota: `idr_quota_user`.
3. Check job: `scontrol show job <jobid>`.
4. IDRIS support: assist@idris.fr (response usually within 1 business day).

---

**Last updated:** August 2026
**Cluster:** Jean Zay (H100, 80GB VRAM)
**Account:** mbp@h100
**Project:** mbp (105524/AD011017614)
**Project name:** EM-VLM4AD