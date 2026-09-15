# nuPlan Data

## Download

Download the nuPlan v1.1 database and map archives from the
[nuPlan download page](https://www.nuscenes.org/nuplan#download). The
[official dataset setup guide](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)
describes the archive layout.

Prepare a nuPlan v1.1 data tree and map package before running evaluation:

```text
<nuplan-data>/nuplan-v1.1/splits/trainval/*.db
<nuplan-data>/nuplan-v1.1/splits/test/*.db
<nuplan-maps>/nuplan-maps-v1.0/*
```

Set the roots:

```bash
export NUPLAN_DATA_ROOT=/path/to/nuplan-data
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps
```

The release configurations use box tracks and map features.

## Benchmark filters

The companion checkout contains one fixed filter for each benchmark split:

| filter | tokens |
| --- | ---: |
| `driverl_val14` | 1118 |
| `driverl_test14_hard` | 272 |
| `driverl_test14_random` | 261 |

Filter files:

```text
nuplan-devkit/nuplan/planning/script/config/common/scenario_filter/driverl_val14.yaml
nuplan-devkit/nuplan/planning/script/config/common/scenario_filter/driverl_test14_hard.yaml
nuplan-devkit/nuplan/planning/script/config/common/scenario_filter/driverl_test14_random.yaml
```

The Random file stores the 261 benchmark tokens and gives the same selection
across database scans.

## Optional database links

Use `DRIVERL_EVAL_DB_LINK_ROOT` to point to directories of symbolic links to
the selected SQLite databases:

```text
<db-link-root>/driverl_val14/
<db-link-root>/driverl_test14_hard/
<db-link-root>/driverl_test14_random/
```

The evaluator also accepts the complete `splits/trainval` and `splits/test`
directories directly.

## Filter check

Run this from the DriveRL root:

```bash
python - <<'PY'
from pathlib import Path
import yaml

root = Path("nuplan-devkit/nuplan/planning/script/config/common/scenario_filter")
expected = {
    "driverl_val14.yaml": 1118,
    "driverl_test14_hard.yaml": 272,
    "driverl_test14_random.yaml": 261,
}
for name, count in expected.items():
    values = yaml.safe_load((root / name).read_text(encoding="utf-8"))["scenario_tokens"]
    assert len(values) == count and len(set(values)) == count, name
    print(f"DRIVERL_FILTER_OK {name} tokens={count}")
PY
```
