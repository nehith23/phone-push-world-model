# Reproducing the submission

Run commands from the repository root. The saved model, five original recordings, processed data, scenarios and results are included. No GPU, paid compute or robot hardware is needed. Training and evaluation use CPU PyTorch and PyBullet in DIRECT mode.

## Verified environment

The reproducibility check used a new Windows x64 Conda environment with Python 3.11.10. The exact package URLs are in [environment-win-64.lock.txt](../environment-win-64.lock.txt), and Python package versions are in [requirements-lock.txt](../requirements-lock.txt).

The simulator is the conda-forge package `pybullet=3.25=py311hbc92ba2_3`. Its Python distribution metadata reports **3.2.5**. These two version labels refer to the installed Conda build; do not replace it with a different PyPI simulator version and assume trajectories remain identical. PyPI does not supply a matching 3.2.5 wheel for this Windows/Python combination.

The original environment used Conda OpenCV 4.9.0 and Matplotlib 3.9.1. The clean environment uses PyPI OpenCV 4.9.0.80 and Matplotlib 3.9.2, because the original Matplotlib pin did not have a usable Windows wheel during the check. It reproduced the data and controller outcomes reported below. Pillow 11.0.0 is explicitly listed for video/GIF reporting.

## Install on Windows

Use a Conda-enabled terminal. Choose a different environment name if `phone-push` already exists; these commands should create a new environment rather than modify another project.

```powershell
conda create -n phone-push --file environment-win-64.lock.txt -y
conda activate phone-push
python -m pip install -r requirements-lock.txt
python -m pip check
```

The clean verification environment was created with Conda and the Python packages were installed with `uv pip`, using the same pinned versions. `pip` subsequently confirmed every lock-file requirement was satisfied; its sandbox-specific temporary-directory cleanup warnings did not change the installed packages. If using uv, activate the Conda environment first and explicitly target its Python executable:

```powershell
uv pip install --python "$env:CONDA_PREFIX\python.exe" -r requirements-lock.txt
```

The Conda lock is platform-specific. Linux was verified separately with Python 3.11 and `uv pip install -r requirements.txt`, where PyBullet 3.2.5 builds from source: artifact checks, all tests, dataset re-extraction and all 48 frozen trials reproduced, with retrained weights within 6e-6 ([evidence](../results/reproduction/linux_verification.json)). macOS has not been verified. [requirements.txt](../requirements.txt) lists the direct Python dependencies, but is not a complete cross-platform simulator lock. Different physics builds or numerical libraries can change trajectories and action choices even when the source code is unchanged.

## Verify and run the frozen controller

```powershell
python scripts/verify_artifacts.py
python -m unittest discover -s tests -v
python scripts/evaluate_precision_frozen.py
python scripts/report_precision_frozen.py
```

The artifact check validates the frozen code and model hashes, all raw-recording hashes, and the summary against all 48 per-controller trial records. The evaluation reuses the existing scenarios and writes their results into `results/precision/fresh/`. It refuses changed frozen code or model weights. Repeating the run reproduces one saved sample; it does not create another independent test.

Expected aggregate results are 24/24 successes and 2.3956 mm mean error for geometry, versus 23/24 and 5.6025 mm for the learned controller. Its failed case is `precision_06`, ending 56.3682 mm from the target. Success is position within 5 mm, with a maximum of 12 pushes and exact simulator pose observations.

## Rebuild the final demo and failure report

```powershell
python scripts/make_precision_demo.py
python scripts/report_precision_failure.py
```

The demo selects the first scenario by index and every learned failure. Both controllers are rendered. The script calls the unchanged frozen controller, enabling rendering through a wrapper; before accepting the video it checks that every resulting state matches the saved record within 1e-12 and that selected action indices are identical. This is a replay, not a new evaluation or a representative random sample. The demo is saved at `results/precision/demo/precision_demo.mp4`.

## Rebuild the dataset and train

Make a separate copy of the repository before these commands. Extraction and training overwrite their outputs, including model checkpoint files. Even when the learned tensors match, serialized checkpoint bytes can differ. Keep the submitted checkpoint and frozen protocols intact.

```powershell
python scripts/stabilize_dataset.py --video-dir data/raw
python scripts/train_stabilized.py
```

Extraction should produce 543 transitions: 424 training, 61 validation and 58 reused-test. Training builds the `old_only` and `plus_pilot` variants, with best validation epochs 259 and 350 respectively. Their artifacts are under `results/stabilized/`.

The final frozen controller uses `results/stabilized/plus_pilot/small_mlp.pt`. The original real-video test split has been inspected during development; these offline scores are development evidence. The fresh simulation protocol is a different kind of evaluation and does not validate physical-session generalization.

To rebuild the development controller comparison in that separate copy:

```powershell
python scripts/control_stabilized.py
python scripts/report_stabilized.py
```

To reproduce the earlier, preserved 15 mm evaluation, use the original submitted checkpoint:

```powershell
python scripts/evaluate_frozen.py
python scripts/report_frozen.py --render-demo
```

## What was verified

| Check | Result |
|---|---|
| Unit checks in a new environment | 21 passed |
| Final controller reruns | All 48 matched success, push count and selected actions |
| Saved versus rerun states | Maximum absolute component difference 0.0 |
| Raw-video re-extraction | All data arrays matched within 1e-8; identifiers and splits matched exactly |
| Retraining both model variants | Maximum parameter difference 0.0; normalization matched within 1e-10 |
| Final video replay | Both controllers in the first and failed cases matched saved states and actions |

See [evaluation and extraction evidence](../results/reproduction/verification.json), [training evidence](../results/reproduction/training_verification.json) and [demo verification](../results/precision/demo/verification.json). Reproduction ran in an isolated project copy so it did not overwrite the submitted evidence. Checkpoint and code hashes were rechecked afterward. The fresh setup confirms this Windows configuration; it is not a promise of bitwise equivalence on every platform.

## If a hash check fails

First compare the file against the submitted version. Git line-ending conversion changes bytes, so the repository includes `.gitattributes` to preserve them. Do not edit protocol hashes merely to make a modified controller pass. Preserve the old experiment and give any changed model or controller its own protocol and new evaluation cases.
