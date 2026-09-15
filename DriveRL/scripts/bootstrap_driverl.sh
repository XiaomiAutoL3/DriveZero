#!/usr/bin/env bash
set -euo pipefail

# Initialize the pinned nuPlan companion, install the shared dependency set and
# both editable packages, then dispatch to the manifest-aware evaluation
# wrapper. Dataset and map files remain external and are never downloaded here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PY="${DRIVERL_PYTHON:-${DRIVERL_EVAL_PYTHON:-}}"
if [[ -z "${PY}" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

NUPLAN_ROOT="${DRIVERL_NUPLAN_ROOT:-${ROOT}/nuplan-devkit}"
if [[ "${NUPLAN_ROOT}" == "${ROOT}/nuplan-devkit" ]]; then
  # Use a caller-provided public URL when the companion is hosted somewhere
  # other than the sibling-relative URL recorded in .gitmodules.
  if [[ -n "${DRIVERL_NUPLAN_REPO_URL:-}" ]]; then
    git config submodule.nuplan-devkit.url "${DRIVERL_NUPLAN_REPO_URL}"
  fi
  if [[ ! -f "${NUPLAN_ROOT}/nuplan/planning/script/run_simulation.py" ]]; then
    git submodule update --init --recursive nuplan-devkit
  fi
fi

if [[ ! -f "${NUPLAN_ROOT}/nuplan/planning/script/run_simulation.py" ]]; then
  printf 'ERROR: companion nuPlan checkout is missing: %s\n' "${NUPLAN_ROOT}" >&2
  printf 'Set DRIVERL_NUPLAN_REPO_URL before bootstrap or DRIVERL_NUPLAN_ROOT to an existing checkout.\n' >&2
  exit 2
fi

if [[ "${DRIVERL_INSTALL:-1}" == "1" ]]; then
  CONSTRAINTS="${DRIVERL_CONSTRAINTS:-${ROOT}/constraints.txt}"
  REQUIREMENTS="${DRIVERL_REQUIREMENTS:-${ROOT}/requirements.txt}"
  [[ -f "${CONSTRAINTS}" ]] || {
    printf 'ERROR: constraints file is missing: %s\n' "${CONSTRAINTS}" >&2
    exit 2
  }
  [[ -f "${REQUIREMENTS}" ]] || {
    printf 'ERROR: requirements file is missing: %s\n' "${REQUIREMENTS}" >&2
    exit 2
  }
  "${PY}" -m pip install -r "${REQUIREMENTS}" -c "${CONSTRAINTS}"
  "${PY}" -m pip install -e "${ROOT}" --no-deps
  "${PY}" -m pip install -e "${NUPLAN_ROOT}" --no-deps
fi

export DRIVERL_EVAL_ROOT="${ROOT}"
export DRIVERL_EVAL_NUPLAN_ROOT="${NUPLAN_ROOT}"
export DRIVERL_EVAL_PYTHON="${PY}"
exec "${ROOT}/scripts/run_driverl_nuplan_eval.sh" "$@"
