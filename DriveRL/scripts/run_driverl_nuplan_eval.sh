#!/usr/bin/env bash
set -euo pipefail

# One public entry point for ordinary/TTS nuPlan evaluation.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MODE="${1:-eval}"
[[ "${MODE}" == eval || "${MODE}" == preflight ]] || { echo "ERROR: mode must be eval or preflight" >&2; exit 2; }

PY="${DRIVERL_EVAL_PYTHON:-${ROOT}/.venv_driverl/bin/python}"
[[ -x "${PY}" ]] || PY="$(command -v python3 || command -v python)"
DRIVERL_ROOT="${DRIVERL_EVAL_DRIVERL_ROOT:-${ROOT}}"
NUPLAN_ROOT="${DRIVERL_EVAL_NUPLAN_ROOT:-${ROOT}/nuplan-devkit}"
DATA_ROOT="${NUPLAN_DATA_ROOT:-}"
MAPS_ROOT="${NUPLAN_MAPS_ROOT:-}"
NUPLAN_DATA_ROOT="${DATA_ROOT}"
NUPLAN_MAPS_ROOT="${MAPS_ROOT}"
MANIFEST="${DRIVERL_EVAL_MANIFEST_PATH:-${ROOT}/release/driverl_checkpoints.yaml}"
RELEASE_ID="${DRIVERL_EVAL_RELEASE_ID:-}"
CONFIG="${DRIVERL_EVAL_CONFIG_PATH:-}"
CHECKPOINT="${DRIVERL_EVAL_CHECKPOINT_PATH:-}"

abspath() {
  [[ "$1" == /* ]] && printf '%s\n' "$1" || printf '%s/%s\n' "${ROOT}" "$1"
}
MANIFEST="$(abspath "${MANIFEST}")"
DRIVERL_ROOT="$(abspath "${DRIVERL_ROOT}")"
NUPLAN_ROOT="$(abspath "${NUPLAN_ROOT}")"
[[ -n "${RELEASE_ID}" ]] || { echo "ERROR: set DRIVERL_EVAL_RELEASE_ID" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "ERROR: missing manifest: ${MANIFEST}" >&2; exit 2; }

manifest_value() {
  "${PY}" - "$MANIFEST" "$RELEASE_ID" "$1" <<'PY'
import sys, yaml
manifest, release_id, field = sys.argv[1:]
data = yaml.safe_load(open(manifest, encoding="utf-8")) or {}
for artifact in data.get("artifacts", []):
    if isinstance(artifact, dict) and artifact.get("id") == release_id:
        value = artifact.get(field)
        if value is not None:
            print(value)
        break
PY
}

if [[ -z "${CONFIG}" ]]; then
  CONFIG="$(manifest_value config_file)"
  [[ -n "${CONFIG}" ]] || case "${RELEASE_ID}" in
    driverl-teacher-u2400) CONFIG=release/configs/driverl_teacher.yaml ;;
    driverl-selfplay-u3800) CONFIG=release/configs/driverl_selfplay.yaml ;;
  esac
fi
if [[ -z "${CHECKPOINT}" ]]; then
  checkpoint_file="$(manifest_value checkpoint_file)"
  if [[ -n "${checkpoint_file}" ]]; then
    candidate="$(abspath "${checkpoint_file}")"
    if [[ -f "${candidate}" ]]; then
      CHECKPOINT="${candidate}"
    elif [[ -n "${DRIVERL_EVAL_CHECKPOINT_ROOT:-}" ]]; then
      CHECKPOINT="${DRIVERL_EVAL_CHECKPOINT_ROOT}/${checkpoint_file}"
      [[ -f "${CHECKPOINT}" ]] || CHECKPOINT="${DRIVERL_EVAL_CHECKPOINT_ROOT}/${checkpoint_file##*/}"
    fi
  fi
fi
[[ -n "${CONFIG}" && -f "$(abspath "${CONFIG}")" ]] || { echo "ERROR: set DRIVERL_EVAL_CONFIG_PATH or use a bundled config" >&2; exit 2; }
[[ -n "${CHECKPOINT}" && -f "$(abspath "${CHECKPOINT}")" ]] || { echo "ERROR: set DRIVERL_EVAL_CHECKPOINT_PATH or provide the release checkpoint" >&2; exit 2; }
CONFIG="$(abspath "${CONFIG}")"
CHECKPOINT="$(abspath "${CHECKPOINT}")"

if [[ "${MODE}" == eval ]]; then
  : "${DATA_ROOT:?Set NUPLAN_DATA_ROOT to the nuPlan data root}"
  : "${MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuPlan maps root}"
fi
for required in \
  "${NUPLAN_ROOT}/nuplan/planning/script/run_simulation.py" \
  "${NUPLAN_ROOT}/nuplan/planning/script/config/simulation/planner/driverl_nuplan_planner.yaml" \
  "${NUPLAN_ROOT}/nuplan/planning/script/config/simulation/ego_controller/driverl_one_stage_controller.yaml" \
  "${NUPLAN_ROOT}/nuplan/planning/script/experiments/simulation/closed_loop_nonreactive_agents_driverl.yaml" \
  "${NUPLAN_ROOT}/nuplan/planning/script/experiments/simulation/closed_loop_reactive_agents_driverl.yaml"; do
  [[ -f "${required}" ]] || { echo "ERROR: companion file is missing: ${required}" >&2; exit 2; }
done

EXPLICIT_LOCAL_GPUS="${LOCAL_GPUS:-}"
REQUESTED_LOCAL_GPUS="${DRIVERL_EVAL_LOCAL_GPUS:-}"
if [[ -n "${REQUESTED_LOCAL_GPUS}" && -n "${EXPLICIT_LOCAL_GPUS}" && "${REQUESTED_LOCAL_GPUS}" != "${EXPLICIT_LOCAL_GPUS}" ]]; then
  echo "ERROR: DRIVERL_EVAL_LOCAL_GPUS=${REQUESTED_LOCAL_GPUS} conflicts with LOCAL_GPUS=${EXPLICIT_LOCAL_GPUS}" >&2
  exit 2
fi
LOCAL_GPUS="${REQUESTED_LOCAL_GPUS:-${EXPLICIT_LOCAL_GPUS:-8}}"
WORKER_MODE="${DRIVERL_EVAL_WORKER_MODE:-ray_distributed}"
TASK_LIST="${DRIVERL_EVAL_TASKS:-val14_nr,val14_r,test14hard_nr,test14hard_r,test14random_nr,test14random_r}"
TTS_ENABLED="${DRIVERL_EVAL_TTS_ENABLED:-0}"
TTS_CANDIDATES="${DRIVERL_EVAL_TTS_NUM_CANDIDATES:-8}"
TTS_SEED="${DRIVERL_EVAL_TTS_SEED:-42}"
DEVICE="${DRIVERL_EVAL_DEVICE:-cuda}"
SAVE_LOGS="${DRIVERL_EVAL_SAVE_SIMULATION_LOGS:-1}"
LIMIT_SCENARIOS="${DRIVERL_EVAL_LIMIT_TOTAL_SCENARIOS:-}"
SCENARIO_TOKENS="${DRIVERL_EVAL_SCENARIO_TOKENS_JSON:-}"
FILTER_OVERRIDE="${DRIVERL_EVAL_SCENARIO_FILTER_OVERRIDE:-}"
DRY_RUN="${DRIVERL_EVAL_DRY_RUN:-0}"
CONTINUE_ON_FAILURE="${DRIVERL_EVAL_CONTINUE_ON_FAILURE:-0}"
PARALLEL_WORKERS="${DRIVERL_EVAL_PARALLEL_WORKERS:-20}"
CPUS_PER_NODE="${CPUS_PER_NODE:-128}"
RAY_GPU_PER_SIM="${DRIVERL_EVAL_GPU_PER_SIMULATION:-0.0625}"
RAY_CPU_PER_SIM="${DRIVERL_EVAL_CPU_PER_SIMULATION:-0.5}"
RAY_TEMP="${DRIVERL_EVAL_RAY_TEMP_DIR:-}"
NNODES="${MLP_WORKER_NUM:-1}"
RANK="${MLP_ROLE_INDEX:-0}"
MASTER_ADDR="${MLP_WORKER_0_HOST:-127.0.0.1}"
RAY_PORT="${MLP_WORKER_0_PORT:-6379}"
RAY_PASSWORD="${RAY_REDIS_PASSWORD:-}"
RAY_DASHBOARD_HOST="${DRIVERL_EVAL_RAY_DASHBOARD_HOST:-127.0.0.1}"
RAY_DASHBOARD="${DRIVERL_EVAL_RAY_INCLUDE_DASHBOARD:-0}"
ROUTE_HORIZON="${DRIVERL_EVAL_ROUTE_GOAL_HORIZON_S:-12.0}"
ROUTE_MIN_SPEED="${DRIVERL_EVAL_ROUTE_GOAL_MIN_SPEED_MPS:-5.0}"
ROUTE_PAIR_MODE="${DRIVERL_EVAL_ROUTE_GOAL_PAIR_MODE:-legacy}"
MAX_ACCEL="${DRIVERL_EVAL_BICYCLE_MAX_ACCELERATION:-4.0}"
MAX_STEERING_RATE="${DRIVERL_EVAL_BICYCLE_MAX_STEERING_RATE:-0.8}"

[[ "${LOCAL_GPUS}" == 1 || "${LOCAL_GPUS}" == 8 ]] || { echo "ERROR: DRIVERL_EVAL_LOCAL_GPUS must be 1 or 8" >&2; exit 2; }
[[ "${WORKER_MODE}" == ray_distributed || "${WORKER_MODE}" == ray_local || "${WORKER_MODE}" == sequential ]] || { echo "ERROR: worker mode must be ray_distributed, ray_local, or sequential" >&2; exit 2; }
[[ "${TTS_ENABLED}" == 0 || "${TTS_ENABLED}" == 1 ]] || { echo "ERROR: DRIVERL_EVAL_TTS_ENABLED must be 0 or 1" >&2; exit 2; }
[[ "${SAVE_LOGS}" == 0 || "${SAVE_LOGS}" == 1 ]] || { echo "ERROR: DRIVERL_EVAL_SAVE_SIMULATION_LOGS must be 0 or 1" >&2; exit 2; }
[[ "${TTS_CANDIDATES}" =~ ^[1-9][0-9]*$ && "${TTS_SEED}" =~ ^[0-9]+$ ]] || { echo "ERROR: TTS candidates and seed must be non-negative integers" >&2; exit 2; }
[[ "${WORKER_MODE}" == sequential || -n "${RAY_TEMP}" || "${MODE}" == preflight ]] || { echo "ERROR: set DRIVERL_EVAL_RAY_TEMP_DIR to a short writable directory" >&2; exit 2; }
[[ "${WORKER_MODE}" != ray_distributed || "${MODE}" == preflight || "${DRY_RUN}" == 1 || -n "${RAY_PASSWORD}" ]] || { echo "ERROR: set RAY_REDIS_PASSWORD for ray_distributed evaluation" >&2; exit 2; }
if [[ "${WORKER_MODE}" != ray_distributed && ( "${NNODES}" != 1 || "${RANK}" != 0 ) ]]; then
  echo "ERROR: ${WORKER_MODE} requires one node with rank 0" >&2; exit 2
fi
[[ "${WORKER_MODE}" != ray_local || "${PARALLEL_WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: parallel workers must be positive" >&2; exit 2; }

export PYTHONPATH="${NUPLAN_ROOT}:${DRIVERL_ROOT}/src${DRIVERL_EVAL_EXTRA_PYTHONPATH:+:${DRIVERL_EVAL_EXTRA_PYTHONPATH}}"
export DRIVERL_ROOT NUPLAN_ROOT NUPLAN_DATA_ROOT NUPLAN_MAPS_ROOT
export NUPLAN_DEVKIT_ROOT="${NUPLAN_ROOT}"
export NUPLAN_EXP_ROOT="${DRIVERL_EVAL_OUTPUT_ROOT:-${ROOT}/output/nuplan_eval}"
export HYDRA_FULL_ERROR=1
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  CUDA_DEVICES="$(seq -s, 0 $((LOCAL_GPUS - 1)))"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
RAY=("${PY}" -m ray.scripts.scripts)

CONFIG_RUN="$(basename "$(dirname "${CONFIG}")")"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}" .pt)"
RUN_ID="${DRIVERL_EVAL_RUN_ID:-driverl_${CONFIG_RUN}_${CHECKPOINT_NAME}_${MLP_TASK_ID:-manual}}"
LOG_DIR="${ROOT}/output/task_logs/${RUN_ID}"
OUT_ROOT="${NUPLAN_EXP_ROOT}/${RUN_ID}"
SUMMARY="${LOG_DIR}/score_summary.tsv"
DONE_FILE="${LOG_DIR}/rank0.done"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"
LOG_FILE="${LOG_DIR}/run.log"
[[ "${RANK}" == 0 ]] || LOG_FILE="${LOG_DIR}/run_node${RANK}.log"

run_preflight() {
  local script="${DRIVERL_ROOT}/scripts/driverl_release_preflight.py"
  [[ -f "${script}" ]] || { echo "PREFLIGHT_ERROR missing_script=${script}" >&2; return 2; }
  local args=("${script}" --manifest "${MANIFEST}" --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --release-id "${RELEASE_ID}" --driverl-root "${DRIVERL_ROOT}" --nuplan-root "${NUPLAN_ROOT}" --device cpu --check-simulation-import)
  "${PY}" "${args[@]}"
}
[[ "${MODE}" != preflight ]] || { run_preflight; exit $?; }
[[ "${DRIVERL_EVAL_SKIP_PREFLIGHT:-0}" == 1 ]] || run_preflight

RAY_SESSION=""
RAY_STARTED=0
ray_session() { [[ -n "${RAY_TEMP}" && -e "${RAY_TEMP}/session_latest" ]] || return 1; readlink -f "${RAY_TEMP}/session_latest" 2>/dev/null; }
capture_ray_session() { for _ in $(seq 1 50); do RAY_SESSION="$(ray_session 2>/dev/null || true)"; [[ -d "${RAY_SESSION}" ]] && return; sleep .2; done; echo "WARNING: Ray session could not be identified" >&2; }
stop_owned_ray() {
  [[ "${RAY_STARTED}" == 1 ]] || return 0
  if [[ -z "${RAY_SESSION}" ]]; then
    echo "WARNING: leaving Ray processes running because the session could not be identified" >&2
    return 0
  fi
  mapfile -t pids < <(ps -eo pid=,args= | awk -v s="${RAY_SESSION}" 'index($0,s){print $1}')
  for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  for _ in $(seq 1 40); do
    alive=0
    for pid in "${pids[@]}"; do kill -0 "${pid}" 2>/dev/null && alive=1; done
    [[ "${alive}" == 0 ]] && break
    sleep .25
  done
  for pid in "${pids[@]}"; do kill -KILL "${pid}" 2>/dev/null || true; done
  RAY_STARTED=0
}

configure_task() {
  local task="$1"
  SIMULATION="closed_loop_nonreactive_agents_driverl"
  [[ "$task" == *_r ]] && SIMULATION="closed_loop_reactive_agents_driverl"
  case "$task" in
    val14_nr|val14_r) protocol="${task##*_}"; LABEL="Val14 ${protocol^^}"; FILTER="driverl_val14"; DB="${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval" ;;
    test14hard_nr|test14hard_r) protocol="${task##*_}"; LABEL="Test14-hard ${protocol^^}"; FILTER="driverl_test14_hard"; DB="${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test" ;;
    test14random_nr|test14random_r) protocol="${task##*_}"; LABEL="Test14-random ${protocol^^}"; FILTER="driverl_test14_random"; DB="${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test" ;;
    *) echo "ERROR: unknown task '${task}'" >&2; return 2 ;;
  esac
  local link="${DRIVERL_EVAL_DB_LINK_ROOT:-}/driverl_${FILTER#driverl_}"
  [[ -n "${DRIVERL_EVAL_DB_LINK_ROOT:-}" && -e "${link}" ]] && DB="${link}"
  [[ -n "${FILTER_OVERRIDE}" ]] && FILTER="${FILTER_OVERRIDE}"
  return 0
}

emit_score() {
  "${PY}" - "$1" "$2" "$3" "$SUMMARY" <<'PY'
from pathlib import Path
import sys, pandas as pd
task, label, out_dir, summary = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
files = sorted((out_dir / "aggregator_metric").glob("*weighted_average_metrics*.parquet"), key=lambda p: p.stat().st_mtime)
if not files: raise SystemExit(f"RESULT_ERROR task={task} reason=no_aggregator_parquet")
df = pd.read_parquet(files[-1]); rows = df[df["num_scenarios"].isna()] if "num_scenarios" in df else df
final = df[(df["scenario"].astype(str) == "final_score") & df["num_scenarios"].notna()] if "scenario" in df else df.iloc[0:0]
row = final.iloc[-1] if not final.empty else None
score = float(row["score"]) if row is not None else float(rows["score"].mean())
total = int(row["num_scenarios"]) if row is not None else len(rows)
runner = pd.read_parquet(out_dir / "runner_report.parquet") if (out_dir / "runner_report.parquet").exists() else None
ok = int(runner["succeeded"].sum()) if runner is not None and "succeeded" in runner else ""
bad = int((~runner["succeeded"]).sum()) if runner is not None and "succeeded" in runner else ""
print(f"RESULT task={task} label='{label}' score={score*100:.2f} successful={ok} failed={bad} total={total} aggregator={files[-1]}")
summary.parent.mkdir(parents=True, exist_ok=True)
if not summary.exists(): summary.write_text("task\tlabel\tscore\tsuccessful\tfailed\ttotal\taggregator\tout_dir\n", encoding="utf-8")
with summary.open("a", encoding="utf-8") as f: f.write(f"{task}\t{label}\t{score*100:.6f}\t{ok}\t{bad}\t{total}\t{files[-1]}\t{out_dir}\n")
PY
}

verify_replay() {
  "${PY}" - "$1" "$2" <<'PY'
from pathlib import Path
import sys, pandas as pd
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.simulation.simulation_log import SimulationLog
task, out = sys.argv[1], Path(sys.argv[2]); report = out / "runner_report.parquet"
if not report.is_file(): raise SystemExit(f"REPLAY_ERROR task={task} missing_runner_report")
runner = pd.read_parquet(report); expected = int(runner["succeeded"].sum())
logs = sorted((out / "simulation_log").rglob("*.msgpack.xz"))
if len(logs) != expected: raise SystemExit(f"REPLAY_ERROR task={task} expected_logs={expected} actual_logs={len(logs)}")
if not logs: raise SystemExit(f"REPLAY_ERROR task={task} no_simulation_logs")
SimulationLog.load_data(file_path=logs[0])
nuboard = sorted(out.glob("*.nuboard"), key=lambda p: p.stat().st_mtime)
if not nuboard or NuBoardFile.load_nuboard_file(nuboard[-1]).simulation_folder != "simulation_log":
    raise SystemExit(f"REPLAY_ERROR task={task} invalid_nuboard")
print(f"REPLAY_READY task={task} logs={len(logs)} sample={logs[0]} nuboard={nuboard[-1]}")
PY
}

run_task() {
  local task="$1"; configure_task "${task}" || return
  [[ -e "${DB}" ]] || { echo "ERROR: scenario DB path does not exist: ${DB}" >&2; return 2; }
  [[ -n "${FILTER_OVERRIDE}" || -f "${NUPLAN_ROOT}/nuplan/planning/script/config/common/scenario_filter/${FILTER}.yaml" ]] || { echo "ERROR: missing scenario filter ${FILTER}" >&2; return 2; }
  local out="${OUT_ROOT}/${SIMULATION}_${task}" run_id="${SIMULATION}_${RUN_ID}_${task}" tts=false
  [[ "${TTS_ENABLED}" == 1 ]] && tts=true
  mkdir -p "${out}"
  local -a filter_args=("scenario_filter.shuffle=false")
  [[ -n "${SCENARIO_TOKENS}" ]] && filter_args+=("scenario_filter.scenario_tokens=${SCENARIO_TOKENS}")
  [[ -z "${SCENARIO_TOKENS}" && -n "${LIMIT_SCENARIOS}" ]] && filter_args+=("scenario_filter.limit_total_scenarios=${LIMIT_SCENARIOS}")
  local -a workers=(worker=sequential distributed_mode=SINGLE_NODE)
  if [[ "${WORKER_MODE}" == ray_distributed ]]; then
    workers=(worker=ray_distributed worker.use_distributed=true "worker.threads_per_node=${CPUS_PER_NODE}" worker.log_to_driver=false distributed_mode=SINGLE_NODE number_of_gpus_allocated_per_simulation=0.1 number_of_cpus_allocated_per_simulation=1)
  elif [[ "${WORKER_MODE}" == ray_local ]]; then
    workers=(worker=ray_distributed worker.use_distributed=false "worker.threads_per_node=${PARALLEL_WORKERS}" worker.log_to_driver=false distributed_mode=SINGLE_NODE "number_of_gpus_allocated_per_simulation=${RAY_GPU_PER_SIM}" "number_of_cpus_allocated_per_simulation=${RAY_CPU_PER_SIM}" max_callback_workers=0 disable_callback_parallelization=true)
  fi
  local -a callbacks=(); [[ "${SAVE_LOGS}" == 0 ]] && callbacks+=(~callback.simulation_log_callback)
  echo "START_EVAL task=${task} label='${LABEL}' db=${DB} filter=${FILTER} output=${out}"
  [[ "${DRY_RUN}" == 1 ]] && { echo "DRY_RUN task=${task}"; return 0; }
  set +e
  "${PY}" "${NUPLAN_ROOT}/nuplan/planning/script/run_simulation.py" \
    "+simulation=${SIMULATION}" scenario_builder=nuplan "scenario_builder.db_files=${DB}" "scenario_filter=${FILTER}" "${filter_args[@]}" "${workers[@]}" "${callbacks[@]}" \
    planner=driverl_nuplan_planner ego_controller=driverl_one_stage_controller \
    "planner.driverl_nuplan_planner.device=${DEVICE}" "planner.driverl_nuplan_planner.config_path=${CONFIG}" "planner.driverl_nuplan_planner.checkpoint_path=${CHECKPOINT}" \
    "planner.driverl_nuplan_planner.route_goal_horizon_s=${ROUTE_HORIZON}" "planner.driverl_nuplan_planner.route_goal_min_speed_mps=${ROUTE_MIN_SPEED}" "planner.driverl_nuplan_planner.route_goal_pair_mode=${ROUTE_PAIR_MODE}" \
    "planner.driverl_nuplan_planner.tts_enabled=${tts}" "planner.driverl_nuplan_planner.tts_num_candidates=${TTS_CANDIDATES}" "planner.driverl_nuplan_planner.tts_seed=${TTS_SEED}" \
    "ego_controller.device=${DEVICE}" "ego_controller.nuplan_bicycle_max_acceleration=${MAX_ACCEL}" "ego_controller.nuplan_bicycle_max_steering_rate=${MAX_STEERING_RATE}" \
    output_dir="${out}" experiment_name="${run_id}" exit_on_failure=true ~main_callback.metric_summary_callback
  local status=$?; set -e
  [[ "${status}" == 0 ]] || { echo "RESULT_ERROR task=${task} status=${status}" >&2; return "${status}"; }
  [[ "${SAVE_LOGS}" == 0 ]] || verify_replay "${task}" "${out}" || return 45
  emit_score "${task}" "${LABEL}" "${out}" || return 46
  echo "END_EVAL task=${task} label='${LABEL}'"
}

export RAY_RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHONPATH}\",\"HYDRA_FULL_ERROR\":\"1\",\"NUPLAN_DATA_ROOT\":\"${NUPLAN_DATA_ROOT}\",\"NUPLAN_MAPS_ROOT\":\"${NUPLAN_MAPS_ROOT}\",\"NUPLAN_EXP_ROOT\":\"${NUPLAN_EXP_ROOT}\"}}"
export RAY_TMPDIR="${RAY_TEMP}" ip_head="${MASTER_ADDR}:${RAY_PORT}" redis_password="${RAY_PASSWORD}" num_nodes="${NNODES}"
{
  echo "suite_id=${RUN_ID} eval_tasks=${TASK_LIST} config=${CONFIG} checkpoint=${CHECKPOINT} local_gpus=${LOCAL_GPUS} worker_mode=${WORKER_MODE} tts=${TTS_ENABLED}"
  if [[ "${DRY_RUN}" != 1 ]]; then
    "${PY}" -c "import torch,sys; n=torch.cuda.device_count(); print('visible_cuda_devices=',n); sys.exit(0 if n >= int('${LOCAL_GPUS}') else 42)"
  fi
  if [[ "${RANK}" == 0 ]]; then
    rm -f "${DONE_FILE}" "${SUMMARY}"
    cleanup() { touch "${DONE_FILE}"; stop_owned_ray; }
    trap cleanup EXIT
    if [[ "${WORKER_MODE}" == ray_distributed && "${DRY_RUN}" != 1 ]]; then
      mkdir -p "${RAY_TEMP}"
      "${RAY[@]}" start --head --node-ip-address="${MASTER_ADDR}" --port="${RAY_PORT}" --redis-password="${RAY_PASSWORD}" --num-cpus="${CPUS_PER_NODE}" --num-gpus="${LOCAL_GPUS}" --include-dashboard="${RAY_DASHBOARD}" --dashboard-host="${RAY_DASHBOARD_HOST}" --temp-dir="${RAY_TEMP}"
      RAY_STARTED=1; capture_ray_session
      cluster_ready=0
      for _ in $(seq 1 180); do
        EXPECTED_GPUS=$((NNODES * LOCAL_GPUS)); EXPECTED_GPUS="${EXPECTED_GPUS}" "${PY}" -c 'import os,ray,sys; ray.init(address="auto",_redis_password=os.environ["redis_password"],log_to_driver=False); sys.exit(0 if ray.cluster_resources().get("GPU",0)>=int(os.environ["EXPECTED_GPUS"]) else 1)' && { cluster_ready=1; break; }
        sleep 5
      done
      [[ "${cluster_ready}" == 1 ]] || { echo "ERROR: Ray cluster did not reach $((NNODES * LOCAL_GPUS)) GPUs" >&2; exit 44; }
    fi
    status=0
    IFS=',' read -r -a tasks <<< "${TASK_LIST// /}"
    for task in "${tasks[@]}"; do set +e; run_task "${task}"; task_status=$?; set -e; if [[ "${task_status}" != 0 ]]; then status="${task_status}"; [[ "${CONTINUE_ON_FAILURE}" == 1 ]] || break; fi; done
    echo "SUITE_DONE suite_id=${RUN_ID} status=${status} summary_file=${SUMMARY}"
    exit "${status}"
  fi
  [[ "${WORKER_MODE}" == ray_distributed ]] || { echo "ERROR: nonzero ranks require ray_distributed" >&2; exit 2; }
  mkdir -p "${RAY_TEMP}"
  joined=0
  for _ in $(seq 1 180); do "${RAY[@]}" start --address="${MASTER_ADDR}:${RAY_PORT}" --redis-password="${RAY_PASSWORD}" --num-cpus="${CPUS_PER_NODE}" --num-gpus="${LOCAL_GPUS}" --temp-dir="${RAY_TEMP}" && { joined=1; break; }; sleep 5; done
  [[ "${joined}" == 1 ]] || { echo "ERROR: failed to join Ray head" >&2; exit 43; }
  RAY_STARTED=1; capture_ray_session
  while [[ ! -f "${DONE_FILE}" ]]; do sleep 30; done
  stop_owned_ray
} 2>&1 | tee "${LOG_FILE}"
