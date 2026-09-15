# DriveRL

DriveRL is the nuPlan inference runtime. The checkout includes the released
policy configurations, inference checkpoints, evaluation wrappers, and a
pinned `nuplan-devkit` companion.

## Install

Run these commands from the repository root with Python 3.11:

```bash
git clone --recurse-submodules <DriveRL-public-url>
cd DriveRL
python -m pip install -r requirements.txt -c constraints.txt
python -m pip install -e . --no-deps
python -m pip install -e nuplan-devkit --no-deps
```

The shared environment contract and CUDA 12.8 PyTorch command are in
[`docs/driverl_environment.md`](docs/driverl_environment.md).

## Release

The release artifact table, checkpoint format, and preflight command are maintained in
[`docs/driverl_repro_eval.md`](docs/driverl_repro_eval.md).

## Evaluation

Prepare the nuPlan database and maps, then follow
[`docs/driverl_nuplan_data.md`](docs/driverl_nuplan_data.md) for the directory
layout and fixed benchmark filters. The ordinary six-task evaluation,
deterministic TTS evaluation, worker settings, and replay
checks are in [`docs/driverl_repro_eval.md`](docs/driverl_repro_eval.md).

The public route contract enables route roadblock correction. The six tasks use
the filters `driverl_val14`, `driverl_test14_hard`, and
`driverl_test14_random`.
