# nuBoard

nuBoard displays nuPlan metric tables, histograms, and scenario replays.

## Start

From the repository root, pass one or more `.nuboard` files or simulation
directories:

```bash
python nuplan/planning/script/run_nuboard.py \
  simulation_path="[/path/to/run.nuboard]"
```

The server listens on `127.0.0.1` by default. For a trusted reverse proxy,
set the bind address and WebSocket origins explicitly:

```bash
export NUPLAN_NUBOARD_ADDRESS=127.0.0.1
export NUPLAN_NUBOARD_WEBSOCKET_ORIGINS=localhost:8080,127.0.0.1:8080
```

The Overview page summarizes metric scores, Histograms shows score
distributions, and Scenarios opens a selected simulation replay.
