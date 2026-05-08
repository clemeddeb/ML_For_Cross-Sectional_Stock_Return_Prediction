#!/usr/bin/env bash
set -euo pipefail

# Submit the full gradient-boosting baseline run to EPFL RCP.
#
# Required:
#   RUNAI_UID       Numeric RCP UID to pass to runai --run-as-uid.
#   RCP_USERNAME    Username owning the home PVC mount.
#
# Optional:
#   RUNAI_JOB_NAME  Defaults to ml-finance-boosting-baseline.
#   RUNAI_IMAGE     Defaults to registry.rcp.epfl.ch/ee559/environment-with-packages:latest.
#   RCP_REPO_DIR    Defaults to this repository path under /home/$RCP_USERNAME.
#   RUNAI_GPU       Defaults to 1.
#   MAX_TRAIN_ROWS  Optional fixed train subsample, e.g. 500000 for a first RCP test.
#   PYTHON_BIN      Defaults to python3 from the RCP image. Set to .venv/bin/python if desired.

RUNAI_UID="${RUNAI_UID:?Set RUNAI_UID to your numeric RCP UID.}"
RCP_USERNAME="${RCP_USERNAME:?Set RCP_USERNAME to your RCP username.}"

RUNAI_JOB_NAME="${RUNAI_JOB_NAME:-ml-finance-boosting-baseline}"
RUNAI_IMAGE="${RUNAI_IMAGE:-registry.rcp.epfl.ch/ee559/environment-with-packages:latest}"
RUNAI_GPU="${RUNAI_GPU:-1}"
RCP_REPO_DIR="${RCP_REPO_DIR:-/home/${RCP_USERNAME}/ML/ML_For_Finance_Project-AxelTurinPlessia-362559-ClementMeddeb-346164}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MAX_TRAIN_ARG=()
if [[ -n "${MAX_TRAIN_ROWS:-}" ]]; then
  MAX_TRAIN_ARG=(--max-train-rows "${MAX_TRAIN_ROWS}")
fi

REMOTE_COMMAND=$(cat <<EOF
set -euo pipefail
cd "${RCP_REPO_DIR}"
mkdir -p outputs/logs outputs/predictions outputs/tables outputs/models/baselines
PYTHON_BIN="${PYTHON_BIN}"
"\${PYTHON_BIN}" scripts/07_train_baselines.py --only-boosting --merge-boosting ${MAX_TRAIN_ARG[*]} 2>&1 | tee outputs/logs/boosting_job.log
EOF
)

runai submit \
  --name "${RUNAI_JOB_NAME}" \
  --run-as-uid "${RUNAI_UID}" \
  --image "${RUNAI_IMAGE}" \
  --gpu "${RUNAI_GPU}" \
  --existing-pvc "claimname=home,path=/home/${RCP_USERNAME}" \
  --command -- bash -lc "${REMOTE_COMMAND}"
