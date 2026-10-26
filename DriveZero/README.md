# DriveZero

DriveZero is a camera-based trajectory planner for autonomous driving. It
learns from the **DriveRL** trajectory teacher during training and runs as a
standalone student at inference time. This repository contains the release
training path, dataset packer, student inference command, and PDM/LQR
evaluation path selected by [`configs/nuplan.yaml`](configs/nuplan.yaml).

The planner consumes four 512 × 960 RGB cameras in the order
`CAM_F0, CAM_B0, CAM_L0, CAM_R0`, camera calibration, a navigation command,
and the current vehicle state. A frozen visual encoder produces patch features;
camera-specific register attention compresses them for trajectory decoding,
while a proposal scorer reads the patch features. The model predicts 64
trajectory proposals, each containing 20 `(x, y, yaw)` poses at 5 Hz, together
with six PDM score logits per proposal.

Training uses a frozen DriveRL teacher, route and off-road goal augmentation,
and both standard and long-horizon trajectory targets. PDM targets supervise 32
randomly selected proposals per sample. Student inference does not require the
DriveRL teacher.

## Installation and external assets

Use Python 3.11 with a CUDA-enabled PyTorch installation. The recommended
baseline uses PyTorch 2.6.0 with CUDA 12.4, torchvision 0.21.0, NumPy 1.26.4,
timm 1.0.22, einops, WebDataset 1.0.2, Ray 2.10.0, PyYAML 6.0.3,
TensorBoard 2.20.0, and PyTurboJPEG 1.8.2. PyTurboJPEG also requires the
system `libturbojpeg` library. OpenCV and tqdm are needed for data packing;
pytest is useful for local checks. The external source trees, model weights,
datasets, and metric caches are not bundled here.

Download the DriveRL checkpoints and DriveVFM weights from the [shared Google
Drive folder](https://drive.google.com/drive/folders/1265Fis9LqvDBDjF6cdF6qjo8854VLheW?usp=sharing).

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==1.26.4 timm==1.0.22 webdataset==1.0.2 ray==2.10.0 \
  einops PyYAML==6.0.3 tensorboard==2.20.0 PyTurboJPEG==1.8.2 opencv-python tqdm pytest
sudo apt-get install libturbojpeg
```

| Asset | Required setup |
| --- | --- |
| DriveRL teacher source | Provide the teacher source tree and add its `src` directory to `PYTHONPATH` for training. |
| DriveRL teacher weights | Provide the teacher checkpoint and set `teacher.checkpoint_path` in `configs/nuplan.yaml`. |
| Visual backbone | Provide the DriveVFM `model_backbone_final.pth` state dict and matching source tree; set `student.model.visual_backbone_weights` and `student.model.visual_backbone_source`. |
| Training scorer | Provide the PDM implementation and its nuPlan devkit; set the scorer root and metric cache paths in the planning configuration. |
| Evaluation stack | Provide NAVSIM v1.1, its NAVTEST metric cache, OpenScene assets, and the nuPlan maps. |
| Training data | Provide packed WebDataset shards and their metadata. |

The teacher checkpoint and visual backbone are supplied separately and are not
copied into this repository.

## Data preparation

The loader reads WebDataset tar shards containing `data.npz` and `meta.json`.
Metadata must include `anchor_token`; the arrays contain agent history and
future states and masks, lane geometry and connectivity, traffic lights, route
points, occupancy, camera calibration, JPEG images, and the navigation command.
Training shards contain 10 Hz states. With `temporal_stride: 2`, the model uses
5 Hz input with 21 history states and 100 future states. Camera images are
aligned to the anchor frame. Evaluation reads the source interval from
`manifest.json` and resamples states to 5 Hz.

The packer starts from scenario shards containing the scene arrays and metadata.
It attaches synchronized nuPlan camera images and calibration together with
OpenScene driving commands; it does not reconstruct the scene arrays from raw
nuPlan logs.

```bash
python -m preprocess.pack_nuplan_dataset \
  --sources /path/to/scenario_shards \
  --db-dir /path/to/nuplan/splits/trainval_sensor \
  --sensor-root /path/to/nuplan/sensor_blobs \
  --command-root /path/to/openscene/navsim_logs/trainval \
  --out-dir data/nuplan/navtrain --frame-time-interval 0.1 --workers 20
```

Run the packer separately for each split and pass that split's source interval.
It writes up to 1,024 samples per shard and produces a manifest. Prepare PDM
metric caches with the matching external evaluator, preserving its
`metadata/*.csv` index and scenario directory layout. Set the resulting cache
roots in `planning.pdm_oracle.metric_cache_paths`.

## Training

Run commands from the repository root. Update the dataset, visual backbone,
DriveRL, scorer, and cache paths in `configs/nuplan.yaml`; relative paths are
resolved from the working directory. The repository's teacher adapter lives
under `teachers/` and expects the configured DriveRL source tree.

```bash
export PYTHONPATH=/path/to/driverl/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1

# Keep Ray socket paths short and caches outside source/data directories.
export TMPDIR=/tmp/drivezero RAY_TMPDIR=/tmp/drivezero NAVSIM_EXP_ROOT=/tmp/drivezero
export XDG_CACHE_HOME=/tmp/drivezero/cache TORCH_HOME=/tmp/drivezero/torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/drivezero/inductor TRITON_CACHE_DIR=/tmp/drivezero/triton
export CUDA_CACHE_PATH=/tmp/drivezero/cuda NUMBA_CACHE_DIR=/tmp/drivezero/numba MPLCONFIGDIR=/tmp/drivezero/mpl
mkdir -p "$TMPDIR"

python trainval.py --config configs/nuplan.yaml
```

The release configuration uses global batch size 256, 10,000 AdamW updates,
cosine learning-rate decay from `2e-4`, and checkpoints every 1,000 updates.
The visual encoder and ego embeddings use BF16 autocast; planning decoders and
losses use FP32. The DriveRL teacher is compiled and frozen. GridMask is active
only during training.

For a single-GPU smoke run, override the batch size, worker count, and update
count:

```bash
python trainval.py --config configs/nuplan.yaml --run-name smoke \
  --override dataset.batch_size=1 dataset.num_workers=0 \
  training.max_updates=1 training.log_interval=1 training.save_interval=1
```

For distributed training, use `torchrun --nproc_per_node=N trainval.py
--config configs/nuplan.yaml`. The global batch size must divide evenly across
ranks, and each rank starts the configured number of PDM CPU workers. Use a
dedicated short temporary root for each job.

Outputs are written under `training.output_root/training.run_name`. Training
resumes from `checkpoint_latest.pt`, including optimizer state. Use `--fresh`
to clear a run before restarting, or choose a new `--run-name` to preserve an
existing run. CLI overrides accept existing YAML fields only.

## Inference and PDM/LQR evaluation

Generate student trajectories with the released checkpoint and an evaluation
dataset:

```bash
python -m evaluation.nuplan_student \
  --checkpoint /path/to/outputs/<run-name>/checkpoint_10000.pt \
  --backbone-weights /path/to/drivevfm/model_backbone_final.pth \
  --dataset-root /path/to/packed/navtest \
  --output outputs/navtest/predictions.pkl --batch-size 1 --num-workers 0
```

Score the protocol-4 file with NAVSIM v1.1 and the NAVTEST cache:

```bash
export NAVSIM_V11=/path/to/navsim-v1.1
export OPENSCENE_DATA_ROOT=/path/to/openscene
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export PYTHONPATH="$NAVSIM_V11"
export TMPDIR=/tmp/drivezero-eval RAY_TMPDIR=/tmp/drivezero-eval NAVSIM_EXP_ROOT=/tmp/drivezero-eval
mkdir -p "$TMPDIR"

python -m navsim.planning.script.run_pdm_score_from_trajectory \
  train_test_split=navtest experiment_name=drivezero_navtest \
  metric_cache_path=/path/to/navtest_metric_cache \
  trajectory_file_path=/absolute/path/to/predictions_navsim.pkl use_lqr=true
```

A complete NAVTEST evaluation must report **12,146 successful scenarios and
0 failures**. A smoke evaluation does not establish the full benchmark score.
Training-time PDM scoring and NAVSIM benchmark scoring are separate external
components.
