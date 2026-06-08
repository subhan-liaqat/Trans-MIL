# =============================================================================
# TransMIL + Hilbert Sorting — Google Colab notebook cells
# Copy EACH "# --- Cell N ---" block into its own Colab cell, then run in order.
# =============================================================================

# --- Cell 1: GPU check ---
# !nvidia-smi


# --- Cell 2: Mount Google Drive ---
# from google.colab import drive
# drive.mount('/content/drive')


# --- Cell 3: Get repo code ---
# Option A (recommended if you pushed latest code to GitHub):
# %cd /content
# !rm -rf Trans-MIL
# !git clone https://github.com/subhan-liaqat/Trans-MIL.git
# %cd /content/Trans-MIL
#
# Option B (if GitHub is outdated — copy your updated project folder from Drive):
# %cd /content
# !rm -rf Trans-MIL
# !cp -r "/content/drive/MyDrive/CPath_Research/Trans-MIL" /content/Trans-MIL
# %cd /content/Trans-MIL


# --- Cell 4: Install dependencies (Colab-safe; does not reinstall torch) ---
# !python -m pip install --upgrade pip
# !python -m pip install -r requirements-colab.txt


# --- Cell 5: Dataset path + config generation ---
import subprocess
from pathlib import Path

DATA_ROOT = "/content/drive/MyDrive/CPath_Research/CAMELYON16_Dataset"
CONFIG = "Camelyon/TransMIL_colab.yaml"

required_repo_files = [
    Path("scripts/configure_colab_data.py"),
    Path("scripts/verify_colab_dataset.py"),
    Path("utils/hilbert_sort.py"),
    Path("datasets/camel_data.py"),
]
missing_repo_files = [str(path) for path in required_repo_files if not path.is_file()]
if missing_repo_files:
    raise FileNotFoundError(
        "Your cloned repo is missing required files:\n"
        + "\n".join(f"  - {path}" for path in missing_repo_files)
        + "\nPush the latest code to GitHub, then re-run Cell 3."
    )

subprocess.run(
    ["python", "scripts/configure_colab_data.py", "--data-root", DATA_ROOT, "--output", CONFIG],
    check=True,
)


# --- Cell 6: Preflight dataset + dataloader test ---
import subprocess

subprocess.run(["python", "scripts/verify_colab_dataset.py", "--config", CONFIG], check=True)


# --- Cell 7: Train ---
import subprocess

subprocess.run(
    ["python", "train.py", "--stage", "train", "--config", CONFIG, "--gpus", "0", "--fold", "0"],
    check=True,
)


# --- Cell 8: Test ---
import subprocess

subprocess.run(
    ["python", "train.py", "--stage", "test", "--config", CONFIG, "--gpus", "0", "--fold", "0"],
    check=True,
)


# --- Cell 9: Pick latest checkpoint ---
from pathlib import Path

ckpts = sorted(Path("logs/Camelyon/TransMIL_colab/fold0").glob("epoch=*.ckpt"))
if not ckpts:
    raise FileNotFoundError("No checkpoint found in logs/Camelyon/TransMIL_colab/fold0")
CKPT = str(ckpts[-1])
print("Using checkpoint:", CKPT)


# --- Cell 10: ROC / PR plots + test predictions ---
import subprocess

subprocess.run(
    [
        "python",
        "scripts/eval_test_and_plot.py",
        "--config",
        CONFIG,
        "--ckpt",
        CKPT,
        "--fold",
        "0",
        "--gpus",
        "0",
        "--positive-class",
        "1",
    ],
    check=True,
)


# --- Cell 11: Interpretability maps ---
import subprocess

subprocess.run(
    [
        "python",
        "scripts/visualize_transmil_interpretability.py",
        "--config",
        CONFIG,
        "--ckpt",
        CKPT,
        "--fold",
        "0",
        "--gpus",
        "0",
        "--coords-dir",
        f"{DATA_ROOT}/coords",
        "--max-slides",
        "20",
        "--target-class",
        "1",
    ],
    check=True,
)


# --- Cell 12: Optional convergence plot (only if metrics.csv exists) ---
import subprocess
from pathlib import Path

metrics_csv = Path("logs/Camelyon/TransMIL_colab/fold0/metrics.csv")
if metrics_csv.is_file():
    subprocess.run(
        [
            "python",
            "scripts/plot_ablation_and_convergence.py",
            "--metrics-csvs",
            str(metrics_csv),
            "--metrics-labels",
            "TransMIL-Hilbert",
            "--metric-name",
            "auc",
            "--out-dir",
            "plots",
        ],
        check=True,
    )
else:
    print(f"Skipping convergence plot; not found: {metrics_csv}")


# --- Cell 13: Copy outputs back to Drive ---
import os
import shutil

source_dir = "/content/Trans-MIL/logs/Camelyon/TransMIL_colab/fold0"
drive_destination_folder = "/content/drive/MyDrive/TransMIL_Outputs"
os.makedirs(drive_destination_folder, exist_ok=True)

base_folder_name = os.path.basename(source_dir)
final_destination_path = os.path.join(drive_destination_folder, base_folder_name)

if os.path.exists(final_destination_path):
    shutil.rmtree(final_destination_path)

shutil.copytree(source_dir, final_destination_path)
print(f"Copied outputs to {final_destination_path}")
