# DriveRL Companion Environment

Use Python 3.11 and the shared versions:

| component | version |
| --- | --- |
| Python | 3.11 |
| NumPy | 1.26.4 |
| PyTorch | 2.7.1 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |
| Ray | 2.51.1 |
| Linux GPU CUDA | 12.8 |

```bash
python3.11 -m venv .venv_driverl
source .venv_driverl/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.7.1" \
  --index-url https://download.pytorch.org/whl/cu128 \
  -c constraints.txt
python -m pip install -r requirements.txt \
  -c constraints.txt
python -m pip install -e . --no-deps
```

For a CPU CI environment, install the same PyTorch version from
`https://download.pytorch.org/whl/cpu`.

Set the two source roots for DriveRL evaluation:

```bash
export DRIVERL_ROOT=/path/to/DriveRL
export NUPLAN_DEVKIT_ROOT=/path/to/nuplan-devkit-public
export PYTHONPATH="$NUPLAN_DEVKIT_ROOT:$DRIVERL_ROOT/src"
python -m nuplan.planning.script.run_simulation --help
```

Ray evaluation uses a short writable session directory:

```bash
export DRIVERL_EVAL_RAY_TEMP_DIR=/path/to/short-runtime-tmp/driverl-ray-n${MLP_ROLE_INDEX:-0}
```
