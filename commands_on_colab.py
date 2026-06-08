# TransMIL + Hilbert Sorting — Google Colab commands
# Paper hyperparameters are kept; only PPEG is replaced by Hilbert sorting.
# Dataset source: https://huggingface.co/datasets/torchmil/Camelyon16_MIL

# --- Cell 1: GPU check ---
!nvidia-smi

# --- Cell 2: Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Cell 3: Clone repo ---
%cd /content
!rm -rf Trans-MIL
!git clone https://github.com/subhan-liaqat/Trans-MIL.git
%cd /content/Trans-MIL

# --- Cell 4: Install dependencies ---
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt

# --- Cell 5: Point to your Drive dataset (EDIT THIS PATH) ---
# Expected layout under DATA_ROOT:
#   coords/              (399 .npy)
#   features_resnet50/   (399 .npy)
#   labels/              (399 .npy)
#   splits.csv           (columns: bag_name, split)
import subprocess

DATA_ROOT = "/content/drive/MyDrive/Camelyon16_MIL"  # <-- change to your Drive folder
CONFIG = "Camelyon/TransMIL_colab.local.yaml"

subprocess.run(
    ["python", "scripts/configure_colab_data.py", "--data-root", DATA_ROOT],
    check=True,
)

# --- Cell 6: Sanity-check one feature bag ---
import numpy as np
from pathlib import Path
import pandas as pd

features_dir = Path(DATA_ROOT) / "features_resnet50"
splits = pd.read_csv(Path(DATA_ROOT) / "splits.csv")
bag_name = str(splits.iloc[0]["bag_name"])
sample = np.load(features_dir / f"{bag_name}.npy")
print("bag_name:", bag_name)
print("feature shape:", sample.shape)  # expect (num_patches, 1024)

# --- Cell 7: Train (paper: CE loss, Lookahead RAdam, lr=2e-4, wd=1e-5, batch=1) ---
import os
os.system(f"python train.py --stage train --config {CONFIG} --gpus 0 --fold 0")

# --- Cell 8: Test on official torchmil test split ---
os.system(f"python train.py --stage test --config {CONFIG} --gpus 0 --fold 0")

# --- Cell 9: Pick latest checkpoint ---
from pathlib import Path

ckpts = sorted(Path("logs/Camelyon/TransMIL_colab.local/fold0").glob("epoch=*.ckpt"))
if not ckpts:
    raise FileNotFoundError("No checkpoint found in logs/Camelyon/TransMIL_colab.local/fold0")
CKPT = str(ckpts[-1])
print("Using checkpoint:", CKPT)

# --- Cell 10: ROC / PR plots + test predictions ---
os.system(
    f"python scripts/eval_test_and_plot.py --config {CONFIG} --ckpt {CKPT} "
    "--fold 0 --gpus 0 --positive-class 1"
)

# --- Cell 11: Interpretability maps (uses coords for Hilbert order) ---
os.system(
    f"python scripts/visualize_transmil_interpretability.py --config {CONFIG} --ckpt {CKPT} "
    f"--fold 0 --gpus 0 --coords-dir {DATA_ROOT}/coords --max-slides 20 --target-class 1"
)

# --- Cell 12: Optional convergence plot (needs metrics.csv from training) ---
!python scripts/plot_ablation_and_convergence.py \
  --metrics-csvs logs/Camelyon/TransMIL_colab.local/fold0/metrics.csv \
  --metrics-labels TransMIL-Hilbert \
  --metric-name auc \
  --out-dir plots

# --- Cell 13: Copy outputs back to Drive ---
import os
import shutil

source_dir = "/content/Trans-MIL/logs/Camelyon/TransMIL_colab.local/fold0"
drive_destination_folder = "/content/drive/MyDrive/TransMIL_Outputs"
os.makedirs(drive_destination_folder, exist_ok=True)

base_folder_name = os.path.basename(source_dir)
final_destination_path = os.path.join(drive_destination_folder, base_folder_name)

if os.path.exists(final_destination_path):
    shutil.rmtree(final_destination_path)

shutil.copytree(source_dir, final_destination_path)
print(f"Copied outputs to {final_destination_path}")
