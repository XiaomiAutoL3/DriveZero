# Reproducible nuPlan Evaluation

Run every command from the DriveRL repository root. Prepare the database and
maps with [`driverl_nuplan_data.md`](driverl_nuplan_data.md), then choose a
release row below.

## Release artifacts

| release id | checkpoint | config |
| --- | --- | --- |
| `driverl-teacher-u2400` | `release/checkpoints/checkpoint_2400.pt` | `release/configs/driverl_teacher.yaml` |
| `driverl-selfplay-u3800` | `release/checkpoints/checkpoint_3800.pt` | `release/configs/driverl_selfplay.yaml` |

The bundled files use the `driverl_weights_only` inference format.

## TTS reward settings

The TTS rollout uses the reward calculators shipped under
`src/driverl/env/engine/reward_calculator/`. The self-play release YAML includes
the deterministic reward defaults in `env.engine_config.domain_randomization`
(`enabled: false`): collision `-1.0`, TTC weight `0.0`, off-road `-2.0`,
cross-lane `1.0`, wrong-way `0.0`, goal threshold `1.5 m`, goal reward `0.5`,
comfort `0.0`, curb clearance `1.0`, and static-speed `1.0`. The remaining
thresholds and disabled components are listed in the same YAML block, so the
TTS selection contract is fully specified by the release config.

## Preflight

```bash
export DRIVERL_EVAL_ROOT="$PWD"
export DRIVERL_EVAL_NUPLAN_ROOT="$PWD/nuplan-devkit"
export DRIVERL_EVAL_RELEASE_ID=driverl-teacher-u2400
export DRIVERL_EVAL_CONFIG_PATH="$PWD/release/configs/driverl_teacher.yaml"
export DRIVERL_EVAL_CHECKPOINT_PATH="$PWD/release/checkpoints/checkpoint_2400.pt"
export DRIVERL_EVAL_PYTHON="$(command -v python)"
bash scripts/run_driverl_nuplan_eval.sh preflight
```

The successful result is `PREFLIGHT_OK`. For self-play, use the second release
row.

## Six-task suite

The ordinary and TTS suites use these six task names:

```text
val14_nr,val14_r,test14hard_nr,test14hard_r,test14random_nr,test14random_r
```

| task | simulation | filter | database root |
| --- | --- | --- | --- |
| `val14_nr` | non-reactive | `driverl_val14` | `$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval` |
| `val14_r` | reactive | `driverl_val14` | `$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval` |
| `test14hard_nr` | non-reactive | `driverl_test14_hard` | `$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/test` |
| `test14hard_r` | reactive | `driverl_test14_hard` | `$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/test` |
| `test14random_nr` | non-reactive | `driverl_test14_random` | `$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/test` |
| `test14random_r` | reactive | `driverl_test14_random` | `$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/test` |

Set the common environment:

```bash
export DRIVERL_EVAL_ROOT="$PWD"
export DRIVERL_EVAL_NUPLAN_ROOT="$PWD/nuplan-devkit"
export DRIVERL_EVAL_RELEASE_ID=driverl-teacher-u2400
export DRIVERL_EVAL_CONFIG_PATH="$PWD/release/configs/driverl_teacher.yaml"
export DRIVERL_EVAL_CHECKPOINT_PATH="$PWD/release/checkpoints/checkpoint_2400.pt"
export NUPLAN_DATA_ROOT=/path/to/nuplan-data
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps
export DRIVERL_EVAL_DEVICE=cuda
export DRIVERL_EVAL_TASKS=val14_nr,val14_r,test14hard_nr,test14hard_r,test14random_nr,test14random_r
export DRIVERL_EVAL_SAVE_SIMULATION_LOGS=1
export DRIVERL_EVAL_ROUTE_GOAL_HORIZON_S=12.0
export DRIVERL_EVAL_ROUTE_GOAL_MIN_SPEED_MPS=5.0
export DRIVERL_EVAL_ROUTE_GOAL_PAIR_MODE=legacy
export DRIVERL_EVAL_BICYCLE_MAX_ACCELERATION=4.0
export DRIVERL_EVAL_BICYCLE_MAX_STEERING_RATE=0.8
```

### One-node GPU run

`DRIVERL_EVAL_LOCAL_GPUS` accepts `1` or `8`. Set a short writable Ray
directory for the host:

```bash
export CUDA_VISIBLE_DEVICES=0
export DRIVERL_EVAL_WORKER_MODE=ray_local
export DRIVERL_EVAL_LOCAL_GPUS=1
export DRIVERL_EVAL_PARALLEL_WORKERS=8
export DRIVERL_EVAL_RAY_TEMP_DIR=/path/to/short-runtime-tmp/driverl-ray-n0
export DRIVERL_EVAL_RUN_ID="${DRIVERL_EVAL_RELEASE_ID}-six-split-h20"
export DRIVERL_EVAL_OUTPUT_ROOT="$PWD/output/nuplan_eval/${DRIVERL_EVAL_RELEASE_ID}/normal"
bash scripts/run_driverl_nuplan_eval.sh eval
```

Use `DRIVERL_EVAL_LOCAL_GPUS=8` and `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
for an eight-GPU host.

### Distributed run

Set these variables on each worker. The example describes twelve workers with
eight GPUs each:

```bash
export DRIVERL_EVAL_WORKER_MODE=ray_distributed
export MLP_WORKER_NUM=12
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST=rank0-host
export MLP_WORKER_0_PORT=6379
export LOCAL_GPUS=8
export DRIVERL_EVAL_LOCAL_GPUS=8
export CPUS_PER_NODE=128
export DRIVERL_EVAL_RAY_TEMP_DIR=/path/to/short-runtime-tmp/driverl-ray-n${MLP_ROLE_INDEX}
export RAY_REDIS_PASSWORD="${RAY_REDIS_PASSWORD:?set a private Ray password}"
export DRIVERL_EVAL_RAY_INCLUDE_DASHBOARD=0
export DRIVERL_EVAL_RAY_DASHBOARD_HOST=127.0.0.1
```

### Ordinary suite

```bash
export DRIVERL_EVAL_TTS_ENABLED=0
export DRIVERL_EVAL_RUN_ID="${DRIVERL_EVAL_RELEASE_ID}-six-split"
export DRIVERL_EVAL_OUTPUT_ROOT="$PWD/output/nuplan_eval/${DRIVERL_EVAL_RELEASE_ID}/normal"
bash scripts/run_driverl_nuplan_eval.sh eval
```

### Deterministic TTS suite

```bash
export DRIVERL_EVAL_TTS_ENABLED=1
export DRIVERL_EVAL_TTS_NUM_CANDIDATES=8
export DRIVERL_EVAL_TTS_SEED=42
export DRIVERL_EVAL_RUN_ID="${DRIVERL_EVAL_RELEASE_ID}-six-split-tts"
export DRIVERL_EVAL_OUTPUT_ROOT="$PWD/output/nuplan_eval/${DRIVERL_EVAL_RELEASE_ID}/tts"
bash scripts/run_driverl_nuplan_eval.sh eval
```

TTS uses horizon `5`, `use_bf16=0`, and
`total_return_switch_margin=0.03`. Candidate zero is the deterministic argmax
proposal; the remaining candidates use the configured seed.

## Result checks

Each task writes an aggregator parquet and `score_summary.tsv`. A replayable
task has one `.msgpack.xz` file per successful scenario, a log readable by
`SimulationLog.load_data()`, and a `.nuboard` file pointing to
`simulation_folder=simulation_log`.

The planner group is `driverl_nuplan_planner`; the controller group is
`driverl_one_stage_controller`. Route roadblock correction is enabled, with
bicycle limits of `4.0 m/s^2` acceleration and `0.8 rad/s` steering rate.
