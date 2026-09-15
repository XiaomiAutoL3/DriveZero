# DriveRL Environment

Use one Python 3.11 environment for DriveRL and the nuPlan companion.

| component | version |
| --- | --- |
| Python | 3.11 |
| NumPy | 1.26.4 |
| PyTorch | 2.7.1 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |
| Ray | 2.51.1 |
| Linux GPU CUDA | 12.8 |

## GPU installation

```bash
python3.11 -m venv .venv_driverl
source .venv_driverl/bin/activate
python -m pip install --upgrade pip

DRIVERL_REPO=/path/to/DriveRL
python -m pip install "torch==2.7.1" \
  --index-url https://download.pytorch.org/whl/cu128 \
  -c "$DRIVERL_REPO/constraints.txt"
python -m pip install -r "$DRIVERL_REPO/requirements.txt" \
  -c "$DRIVERL_REPO/constraints.txt"
python -m pip install -e "$DRIVERL_REPO" --no-deps
python -m pip install -e "$DRIVERL_REPO/nuplan-devkit" --no-deps
```

Verify the selected PyTorch build:

```bash
python - <<'PY'
import torch
assert torch.__version__.split("+", 1)[0] == "2.7.1"
assert torch.version.cuda == "12.8"
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
PY
```

CI installs the same version from the PyTorch CPU index.

## Source selection

Select the DriveRL and companion source roots for nuPlan evaluation:

```bash
PYTHONPATH="/path/to/DriveRL/nuplan-devkit:/path/to/DriveRL/src" \
  python -m nuplan.planning.script.run_simulation --help
```

## Ray and NuBoard

For distributed evaluation, provide a private `RAY_REDIS_PASSWORD`. Ray's
dashboard is disabled by default. Set `DRIVERL_EVAL_RAY_INCLUDE_DASHBOARD=1`
for local diagnostics and keep its host at `127.0.0.1`.

Set a short writable directory for Ray session files:

```bash
export DRIVERL_EVAL_RAY_TEMP_DIR=/path/to/short-runtime-tmp/driverl-ray-n${MLP_ROLE_INDEX:-0}
```

NuBoard uses loopback by default. A reverse proxy can provide an explicit
`NUPLAN_NUBOARD_WEBSOCKET_ORIGINS` list.
