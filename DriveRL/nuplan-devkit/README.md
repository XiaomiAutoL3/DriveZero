# nuPlan Devkit for DriveRL

This repository supplies the nuPlan database, map, scenario-builder,
simulation, metrics, serialization, NuBoard, and DriveRL WebDataset exporter
runtime. DriveRL policy code and released checkpoints are in the DriveRL
repository.

## Install

Use Python 3.11:

```bash
python -m pip install -r requirements.txt -c constraints.txt
python -m pip install -e . --no-deps
```

The shared CUDA/PyTorch setup is documented in this repository's
`docs/driverl_environment.md`.

## DriveRL integration

Keep DriveRL and this repository as sibling checkouts and select both source
roots for an integration process:

```bash
export DRIVERL_ROOT=/path/to/DriveRL
export NUPLAN_DEVKIT_ROOT=/path/to/nuplan-devkit-public
export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$DRIVERL_ROOT/src"
```

The DriveRL wrapper accepts the same roots through
`DRIVERL_EVAL_NUPLAN_ROOT` and `DRIVERL_EVAL_DRIVERL_ROOT`. The integration
groups are:

```text
planner:     driverl_nuplan_planner
ego control: driverl_one_stage_controller
ordinary:    closed_loop_nonreactive_agents_driverl
reactive:    closed_loop_reactive_agents_driverl
```

Policy config, profile, checkpoint, database, and map paths are supplied by the
DriveRL release command. Run evaluation from the DriveRL checkout.

## Evaluation filters

The companion includes the fixed DriveRL benchmark filters:

| file | tokens |
| --- | ---: |
| `driverl_val14.yaml` | 1118 |
| `driverl_test14_hard.yaml` | 272 |
| `driverl_test14_random.yaml` | 261 |

They live under
`nuplan/planning/script/config/common/scenario_filter/`.

## WebDataset export

The exporter accepts one SQLite database or a directory of databases and writes
tar shards plus `manifest.json`:

```bash
python nuplan/planning/script/export_driverl_webdataset.py \
  --input /path/to/nuplan-v1.1/splits/trainval \
  --output /path/to/data/nuplan/standard_train_base \
  --map-root /path/to/nuplan-maps \
  --sample-rate-hz 5 \
  --history-frames 21 \
  --future-frames 100
```

The input database supplies lidar, ego, object, and scene metadata. Use
`standard_train_base` for the base export and `standard_train_supplement` for
a separate supplement export; add `--camera-alignment` for camera-aligned
supplement frames. Map export uses the selected map package; camera-aligned
export uses the camera records and blobs in the database tree.

## External roots

Set these roots for simulation and exporter commands:

```bash
export NUPLAN_DATA_ROOT=/path/to/nuplan-data
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps
export NUPLAN_EXP_ROOT=$PWD/output/nuplan
```
